import os
import numpy as np
from scipy.interpolate import UnivariateSpline
import cv2
import h5py
from defisheye import Defisheye
import tqdm
import time

from typing import List, Tuple, Dict, Any

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
                         resolution:Tuple[int,int]=(256,256),
                         metadata:Dict[str,Any]={}):
    
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

    # load the video data, which is split into chunks. Each chunk has a video file and a log file
    # first, determine the number of video files
    n_videos = 0
    total_frames = 0
    while os.path.exists(os.path.join(data_folder_path, VIDEO_PREFIX + str(n_videos) + VIDEO_SUFFIX)):
        # if n_videos == 1: # for testing purposes
        #     break
        cap = cv2.VideoCapture(os.path.join(data_folder_path, VIDEO_PREFIX + str(n_videos) + VIDEO_SUFFIX))
        total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # load the first frame and create the defisheye object
        if defisheye is None:
            ret, frame = cap.read()
            defisheye = Defisheye(frame, **DEWARPING_PARAMS)
            defisheye.calculate_conversions()

        cap.release()
        n_videos += 1

    print(f"Found {n_videos} video files with a total of {total_frames} frames")


    assert n_videos > 0, "No video files found in the data folder"
    
    # open the hdf5 file:
    with h5py.File(os.path.join(data_folder_path, save_name), "w") as save_file:

        # create datasets:
        image_dataset = save_file.create_dataset(name="images", 
                                              shape=(total_frames, resolution[0], resolution[1], 3), 
                                              chunks = (1, resolution[0], resolution[1], 3),
                                              dtype=np.uint8,
                                              compression=9)
        
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
                                                dtype=np.bool)
        
        time_dataset = save_file.create_dataset(name="time",
                                                shape=(total_frames,),
                                                chunks = (1,),
                                                dtype=np.float32)
        
        # add additional metadata:
        save_file.attrs.update(metadata)
        
        # loop through the video files and load the data
        progress_bar = tqdm.tqdm(total=total_frames)
        frame_num = 0
        start_time = 0
        pixel_sum = np.zeros(3)
        pixel_square_sum = np.zeros(3)
        count = 0
        for video_idx in range(n_videos):
            video_file = os.path.join(data_folder_path, VIDEO_PREFIX + str(video_idx) + VIDEO_SUFFIX)
            log_file = os.path.join(data_folder_path, VIDEO_LOG_PREFIX + str(video_idx) + VIDEO_LOG_SUFFIX)

            # load the video log. It is a csv file with the following columns: time,frame_number,path_to_video
            with open(log_file, "r") as f:
                video_log = f.read().strip().split("\n")
            
            video_data = []
            for line in video_log:
                elements = line.split(",")
                if len(elements) < 3:
                    raise Exception(f"Video log file has incorrect number of elements in line: {line}")
                data = [float(elements[0]), int(elements[1]), elements[2]]
                video_data.append(data)

            # load the video data
            cap = cv2.VideoCapture(video_file)

            # make sure the number of frames in the video is the same as the number of frames in the log
            assert len(video_data) == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "Number of frames in video and log do not match!"

            # loop through the video data and extract the frames 
            for data in video_data:
                
                assert data[1] == frame_num, "Frame numbers are not sequential!"
                frame_time = data[0]

                if start_time == 0:
                    start_time = frame_time

                # read the frame
                ret, frame = cap.read()

                # get the current position of the printer:
                printer_pos = printer_data.get_position(frame_time)

                # adjust the printer position to be relative to the center of the printer
                printer_pos = (printer_pos[0] - printer_center[0], printer_pos[1] - printer_center[1])

                force = force_data.get_force(frame_time)

                # check if the time is valid:
                valid = force_data.valid_time(frame_time) and printer_data.valid_time(frame_time)
                
                image = defisheye.unwarp(frame)
                # crop the image
                image = image[CROP[0]:CROP[1], CROP[2]:CROP[3]]

                # resize the image
                image = cv2.resize(image, resolution)
                
                # save the data to the hdf5 file
                image_dataset[frame_num,:,:,:] = image
                force_dataset[frame_num] = force
                position_dataset[frame_num,:] = printer_pos
                valid_dataset[frame_num] = valid
                time_dataset[frame_num] = frame_time
                pixel_sum += np.sum(image.astype(np.float64), axis=(0,1))
                pixel_square_sum += np.sum(image.astype(np.float64)**2, axis=(0,1))
                count += 1
                
                frame_num += 1
                progress_bar.update(1)
                
            cap.release()


        pixel_mean = pixel_sum / (count * resolution[0] * resolution[1])
        pixel_var = pixel_square_sum / (count * resolution[0] * resolution[1]) - pixel_mean**2
        pixel_std = np.sqrt(pixel_var)

        save_file.attrs["pixel_mean"] = pixel_mean
        save_file.attrs["pixel_std"] = pixel_std

        print('pixel mean:', pixel_mean)
        print('pixel std:', pixel_std)
        print('pixel var:', pixel_var)

        save_file.attrs["average_force"] = np.mean(force_dataset[:])
            
if __name__ == "__main__":
    data_folder_path = "/home/aigeorge/research/BeadSight/data/initial_test_34"
    save_name = "processed_data.hdf5"
    printer_center = (150.5, 109.5)
    meta_data = {
        "resolution": (256, 256),
        "fps": 30,
        "contact_radius": 6.25,
        "sensor_size": (41, 41),
        "force_unit": "g",
        "dist_unit": "mm",
    }
    resolution = (256, 256)
    convert_data_to_hdf5(data_folder_path, save_name, printer_center, resolution, meta_data)