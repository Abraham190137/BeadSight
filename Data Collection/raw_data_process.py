import os
import numpy as np
from scipy.interpolate import UnivariateSpline
import cv2
import h5py
from defisheye import Defisheye
import tqdm
import time
import torch
from torchvision.transforms import functional as F

from multiprocessing import Pool

from typing import List, Tuple, Dict, Any, NamedTuple

class VideoMetaData(NamedTuple):
    path: str
    times: List[float]
    start_frame: int
    num_frames: int

def get_video_metadata(video_log_paths:List[str], 
                       video_paths:List[str], 
                       start_time:float, 
                       end_time:float) -> List[VideoMetaData]:
    
    assert len(video_log_paths) == len(video_paths), "Number of video log paths and video paths do not match"
    # get the video meta data:
    
    all_times = []
    last_time = 0
    all_frame_nums = []
    last_frame_num = -1
    for video_log_path, video_path in zip(video_log_paths, video_paths):
        # load the video data:
        with open(video_log_path, "r") as f:
            video_log = f.read().strip().split("\n")

        video_name = os.path.basename(video_path)
        times = []
        frame_nums = []
        for line in video_log:
            elements = line.split(",")
            if len(elements) < 3:
                raise Exception(f"Video log file has incorrect number of elements in line: {line}")
            frame_time = float(elements[0])
            frame_num = int(elements[1])
            times.append(frame_time)
            frame_nums.append(frame_num)

            assert frame_time > last_time, "Times are not sequential"
            last_time = float(elements[0])

            assert frame_num == last_frame_num + 1, "Frame numbers are not sequential"
            last_frame_num += 1

            # make sure the recorded path is the same as the path we are using
            assert elements[2].endswith(video_name), "Video log file does not match video file"
        
        all_times.append(times)
        all_frame_nums.append(frame_nums)

        # make sure the number of frames in the video is the same as the number of frames in the log
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Could not open video file: {video_path}"
        assert len(times) == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "Number of frames in video and log do not match!"
        cap.release()

    print(f"Fount {len(video_paths)} videos, with a total of {sum([len(times) for times in all_times])} frames")
        
    # adjust the meta data for start/end times
    
    # first, adjust the start and end time to make them in global time
    start_time += all_times[0][0]
    
    if end_time == -1: # if end time is -1, set it to the end of the last video
        end_time = all_times[-1][-1]
    else:
        end_time += all_times[0][0]

    # loop through the video meta data and adjust the times
    meta_data_list = []
    total_frames = 0
    for i in range(len(video_paths)):
        if all_times[i][0] > end_time:
            continue # skip this video, as it starts after the time range
        elif all_times[i][0] < start_time:
            start_idx = 0 # start from the beginning
        else:
            start_idx = np.searchsorted(all_times[i], start_time, side="left")

        if all_times[i][-1] < start_time:
            continue # skip this video, as it ends before the time range
        elif all_times[i][-1] > end_time:
            end_idx = len(all_times[i]) # go to the end
        else:
            end_idx = np.searchsorted(all_times[i], end_time, side="right")

        meta_data = VideoMetaData(path = video_paths[i], 
                                  times = all_times[i][start_idx:end_idx], 
                                  start_frame = start_idx,
                                  num_frames = end_idx - start_idx)
        
        meta_data_list.append(meta_data)

    assert len(meta_data_list) > 0, "No videos found in the time range"
    print(f"Using {len(meta_data_list)} videos, with a total of {total_frames} frames")

    return meta_data_list

    

DEWARPING_PARAMS = {
    "fov": 180,
    "pfov": 120,
    "xcenter": None,
    "ycenter": None,
    "radius": None,
    "angle": 0,
    # "dtype": "equalarea",
    "dtype": "linear",
    "format": "fullframe"
}

class ForceData:
    def __init__(self, file_path:str, cutoff_low:float=10, cutoff_high:float=50):
        # load the force data, which is a csv file with the following columns: time,force
        self.data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        self.start_time = self.data[0,0]
        self.end_time = self.data[-1,0]

        # convert the force data into a spline
        self.force_spline = UnivariateSpline(self.data[:,0], self.data[:,1], s=0)

    def get_force(self, time:float) -> float:
        """
        Get the force at a given time. If the time is outside the range of the force data,
        return None
        """
        if not self.valid_time(time):
            return 0.0
        
        return self.force_spline(time)
    
    def valid_time(self, time:float) -> bool:
        """
        Check if the time is within the range of the force data
        """
        return time >= self.start_time and time <= self.end_time
    

class PrinterData:
    def __init__(self, file_path:str):
        # load the printer data, which is a csv file with the following columns: time,pos_x,pos_y,message. It is seperated by \r
        self.printer_data = []
        with open(file_path, "r", newline='') as f:
            file_contents = f.read().strip()

        for line in file_contents.split("\r"):
            elements = line.split(",")
            if len(elements) != 4:
                raise Exception(f"Printer log file has incorrect number of elements in line: {line}")
            self.printer_data.append([float(elements[0]), float(elements[1]), float(elements[2]), elements[3]])
        
        self.last_idx = 0
        self.start_time = self.printer_data[0][0]
        self.end_time = self.printer_data[-1][0]

    def get_position(self, time:float) -> tuple[float, float]:
        """
        Get the position of the printer at a given time. If the time is outside 
        the range of the printer data, or the printer was in the calibration phase,
        return -1 -1.
        """
        if time < self.start_time or time > self.end_time:
            return -1.0, -1.0
        
        # find the index of the data point that is closest to the time
        # Because we go sequentially through the data, we can start from the last
        # index and go forward, incrementing the index when the time is greater than
        # the time of the next data point.

        # The time should never go backwards, so if it does for some reason, we 
        # will start from the beginning of the data
        idx = self.last_idx
        if time < self.printer_data[idx][0]:
            idx = 0
            print("Printer time went backwards, starting from the beginning of the data")

        while time > self.printer_data[idx + 1][0]:
            idx += 1

        if not (idx == self.last_idx or idx == self.last_idx + 1):
            print("Printer incremented index by more than 1")

        self.last_idx = idx
       
        return self.printer_data[idx][1], self.printer_data[idx][2]
    
    def valid_time(self, time:float) -> bool:
        """
        Check if the time is within the range of the printer data
        """
        pos_x, pos_y = self.get_position(time)

        return pos_x != -1.0 and pos_y != -1.0
  

def convert_data_to_hdf5(data_folder_path:str, 
                         save_name:str, 
                         printer_center:Tuple[float,float], # can be determined from the printer log
                         start_time:float=0,
                         end_time:float=-1,
                         resolution:Tuple[int,int]=(256,256),
                         metadata:Dict[str,Any]={},
                         compression_level:int=0,
                         batch_size:int = 256):
    
    # first, we need to load the forces and make a spline:

    FORCE_FILE = "force_log.txt"
    PRINTER_FILE = "printer_log.txt"
    VIDEO_PREFIX = "video_"
    VIDEO_SUFFIX = ".avi"
    VIDEO_LOG_PREFIX = "video_log_"
    VIDEO_LOG_SUFFIX = ".txt"
    CROP = (142, 892, 120, 870) # crop the image to remove the edges. Top, bottom, left, right


    # load the force data, using the ForceData class:
    force_data = ForceData(file_path=os.path.join(data_folder_path, FORCE_FILE))

    # load the printer data, using the PrinterData class:
    printer_data = PrinterData(file_path=os.path.join(data_folder_path, PRINTER_FILE))

    defisheye: Defisheye = None

    # get the video meta data:
    video_log_paths = []
    video_paths = []
    max_files = 1000
    for i in range(max_files):
        video_log_path = os.path.join(data_folder_path, VIDEO_LOG_PREFIX + str(i) + VIDEO_LOG_SUFFIX)
        video_path = os.path.join(data_folder_path, VIDEO_PREFIX + str(i) + VIDEO_SUFFIX)

        if os.path.exists(video_log_path):
            assert os.path.exists(video_path), f"Video log file {video_log_path} exists, but video file {video_path} does not"
            video_log_paths.append(video_log_path)
            video_paths.append(video_path)
        else:
            break
    else:
        raise Exception(f"More than {max_files} video files found in the data folder")
    
    assert len(video_log_paths) > 0, "No video files found in the data folder"
    
    meta_data_list = get_video_metadata(video_log_paths, video_paths, start_time, end_time)
    total_frames = sum([meta_data.num_frames for meta_data in meta_data_list])

    # load the first video to initialize the defisheye object
    cap = cv2.VideoCapture(meta_data_list[0].path)
    _, frame = cap.read()
    defisheye = Defisheye(frame, **DEWARPING_PARAMS)
    defisheye.calculate_conversions()
    cap.release()
    
    # open the hdf5 file:
    with h5py.File(os.path.join(data_folder_path, save_name), "w") as save_file:

        # create datasets:
        if compression_level > 0:
            image_dataset = save_file.create_dataset(name="images", 
                                                shape=(total_frames, resolution[0], resolution[1], 3), 
                                                chunks = (1, resolution[0], resolution[1], 3),
                                                dtype=np.uint8,
                                                compression=compression_level)
        else:
            image_dataset = save_file.create_dataset(name="images", 
                                                shape=(total_frames, resolution[0], resolution[1], 3), 
                                                chunks = (1, resolution[0], resolution[1], 3),
                                                dtype=np.uint8)
        
        force_dataset = save_file.create_dataset(name="forces",
                                              shape=(total_frames,),
                                              chunks = (1,),
                                              dtype=np.float32)
        
        position_dataset = save_file.create_dataset(name="position",
                                                shape=(total_frames, 2),
                                                chunks = (1, 2),
                                                dtype=np.float32)
        
        valid_dataset = save_file.create_dataset(name="valid",
                                                shape=(total_frames,),
                                                chunks = (1,),
                                                dtype=bool)
        
        time_dataset = save_file.create_dataset(name="time",
                                                shape=(total_frames,),
                                                chunks = (1,),
                                                dtype=np.float32)
        
        images = []
        forces = []
        positions = []
        valids = []
        times = []
        
        # add additional metadata:
        save_file.attrs.update(metadata)
        
        # loop through the video files and load the data
        progress_bar = tqdm.tqdm(total=total_frames)
        frame_count = 0
        pixel_sum = np.zeros(3)
        pixel_square_sum = np.zeros(3)
        for video_meta_data in meta_data_list:            
            # load the video data
            cap = cv2.VideoCapture(video_meta_data.path)
            
            # set the start frame
            if video_meta_data.start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, video_meta_data.start_frame)

            # loop through the video data and extract the frames 
            for frame_time in video_meta_data.times:

                # read the frame
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Could not read frame from video")
                frame_count += 1
                # get the current position of the printer:
                printer_pos = printer_data.get_position(frame_time)

                # adjust the printer position to be relative to the center of the printer
                printer_pos = (printer_pos[0] - printer_center[0], printer_pos[1] - printer_center[1])

                force = force_data.get_force(frame_time)

                # check if the time is valid:
                valid = force_data.valid_time(frame_time) and printer_data.valid_time(frame_time)

                # add the data to the lists. We will process the data in batches
                images.append(frame)
                forces.append(force)
                positions.append(printer_pos)
                valids.append(valid)
                times.append(frame_time)

                # if its the last frame, we need to process the batch
                if frame_count == total_frames:
                    batch_size = len(images)

                # process the batch, using cuda to speed up image processing
                if len(images) == batch_size:
                    with torch.no_grad():
                        images = torch.from_numpy(np.array(images))
                        images = images.cuda()
                        images = defisheye.unwarp(images)
                        images = images[:, CROP[0]:CROP[1], CROP[2]:CROP[3]]

                        # Resize images. Need to rearange n, h, w, c -> n, c, h, w
                        images = images.permute(0, 3, 1, 2)
                        images = F.resize(images, resolution)
                        images = images.permute(0, 2, 3, 1)

                        # statistics for normalization
                        pixel_sum += torch.sum(images.float(), dim=(0,1,2)).cpu().numpy()
                        pixel_square_sum += torch.sum(images.float()**2, dim=(0,1,2)).cpu().numpy()

                        images = images.cpu().numpy()
                    
                    # save the batch:
                    image_dataset[frame_count - batch_size:frame_count,:,:,:] = images
                    force_dataset[frame_count - batch_size:frame_count] = np.array(forces)
                    position_dataset[frame_count - batch_size:frame_count,:] = np.array(positions)
                    valid_dataset[frame_count - batch_size:frame_count] = np.array(valids)
                    time_dataset[frame_count - batch_size:frame_count] = np.array(times)
                    
                    # reset the batch lists
                    images = []
                    forces = []
                    positions = []
                    valids = []
                    times = []
                
                progress_bar.update(1)
            
            # make sure that we have processed all the frames (exept the last video)
            ret, _ = cap.read()
            if frame_count < total_frames:
                assert not ret, "Video has more frames than expected"
            cap.release()

        assert len(images) == 0, "Images were not processed correctly"

        pixel_mean = pixel_sum / (frame_count * resolution[0] * resolution[1])
        pixel_var = pixel_square_sum / (frame_count * resolution[0] * resolution[1]) - pixel_mean**2
        pixel_std = np.sqrt(pixel_var)

        save_file.attrs["pixel_mean"] = pixel_mean
        save_file.attrs["pixel_std"] = pixel_std

        print('pixel mean:', pixel_mean)
        print('pixel std:', pixel_std)
        print('pixel var:', pixel_var)

        save_file.attrs["average_force"] = np.mean(force_dataset[:])
            
if __name__ == "__main__":

    meta_data = {
        "resolution": (256, 256),
        "fps": 30,
        "contact_radius": 6.25,
        "sensor_size": (41, 41),
        "force_unit": "g",
        "dist_unit": "mm",
    }

    convert_data_to_hdf5(data_folder_path = "/home/abraham/BeadSight/data/12_hr_100_0", 
                         save_name = "processed_data.hdf5",
                         printer_center = (150.5, 109.5), 
                         start_time=60*60, # seconds
                         end_time=60*60*3, # seconds
                         resolution = (256, 256), 
                         metadata = meta_data, 
                         compression_level=0, 
                         batch_size=256)