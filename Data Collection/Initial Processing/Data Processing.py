import numpy as np
import cv2
import scipy.interpolate
import matplotlib.pyplot as plt
import pickle
import json
from typing import List, Dict, Tuple, Any
from scipy.interpolate import UnivariateSpline
import os

### IMPORTANT ### 
# The code assumes that in the video, as the x location increases the finger moves to the right, and as the 
# y location increases the finger moves down. If this is not the case, the code will need to be modified.

"""
This code does the initial processing to create pressure maps from the scale data (WEIGHTS_FILE), and
processes the G-Code file (info.json). It splits the video into individual tests (based on the scale data), 
and generates the pickle file (which can be used to generate the pressure map) for each test.
"""

# Helper function to read the time stamp from the wieght data
def time_stamp_to_seconds(time_stamp:str) -> float:
    time_stamp = time_stamp.split(':')
    seconds = float(time_stamp[2])
    seconds += float(time_stamp[1])*60
    seconds += float(time_stamp[0])*3600
    return seconds

# Load in the wieght data. 
def load_force_data(force_data_file:str) -> np.ndarray:
    """
    :param force_data_file: The file containing the wieght data (recoreded from the scale)
    :return: A numpy array of the force data. Each row is a force measurement, and the first column is the force in grams,
                and the second column is the time in seconds.
    """
    force_data_txt: str = open(WIEGHTS_FILE, 'r').read().split('\n')
    force_data: List[List[float]] = []
    for line in force_data_txt:
        split_line: List[str] = line.split(',')
        if len(split_line) != 3:
            continue
        force: float = float(split_line[0])
        seconds: float = time_stamp_to_seconds(split_line[2])
        force_data.append((force, seconds))
    
    return np.array(force_data)

# Helperfunction to have a user select a frame from the video
def select_frame(cap):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    start_time = input("Enter start time, after the calibration presses, but before the first test: ")
    cv2.namedWindow("frame")
    start_frame = int(float(start_time)*fps)
    print(start_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fc = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("frame", frame)
        cv2.waitKey(1) 
        command = input("frame " + str(fc) +". To stop, type 'stop'. \n"
                        + "To move to a different frame, enter frame number."
                        + "To move to next fame, press enter. ")
        if command == "stop":
            break
        if command != "":
            try:
                fc = int(command)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fc)
            except:
                print("Invalid input. Proceeding to next frame. ")
        else:
            fc += 1

    time_stamp = input('Plese enter the timestamp: ')
    cv2.destroyAllWindows()
    return fc, time_stamp

# Generates the info dictionary (and saves as a json file)
def geerate_info_file(video_file, gcode_file, center_point, dir) -> Dict[str, Any]:
    """
    :param video_file: The video file of the test
    :param gcode_file: The gcode file of the test
    :param center_point: The center point of the sensor in g-code
    :param dir: The directory to save the info file in
    :return: The info dictionary
    """
    cap = cv2.VideoCapture(video_file)
    print('Select start frame, the frame to start the first push.')
    start_frame, time_stamp = select_frame(cap)
    cap.release()


    points = []
    with open(gcode_file, 'r') as gcode_file:
        gcode = gcode_file.read().split('\n')

    for line in gcode:
        # The gcode command lines are of the form "G1 Z## X## Y##"
        # Note: The first press is NOT recorded as a test, but is included.
        # It's used to find the center of the sensor of calibration
        if (not "G1" in line) or ("calibration" in line) or ("move up" in line):
            continue
        x_idx = line.rfind("X")
        y_idx = line.rfind("Y")
        z_idx = line.rfind("Z")

        x = float(line[x_idx+1:y_idx-1])
        y = float(line[y_idx+1:z_idx-1])

        point = (x, y)
        if len(points) == 0 or points[-1] != point:
            points.append(point)    

    save_dict = {}
    save_dict['start_frame'] = start_frame
    save_dict['start_timestamp'] = time_stamp
    save_dict['points'] = points
    save_dict['video_name'] = video_file.split('/')[-1]
    save_dict['center_point'] = center_point

    with open(dir + 'info.json', 'w') as json_file:
        json.dump(save_dict, json_file)
    
    return save_dict

if __name__ == "__main__":
    # Inputs, need to be changed for each run:
    # Directory of the data, video, and info file
    DIR = "C:/Research/Tactile Sensor/Tactile_Sensor/Data Collection/data/7-12_test_2/"

    WIEGHTS_FILE = DIR + "weight_readings_7-12_5.csv"
    VIDEO_FILE = DIR + "2023-07-12 08-40-59.mp4"
    GCODE_FILE = DIR + "data_collection_3_7-12.g"
    
    # Center point of the sensor in g-code.
    CENTER_POINT = [160, 85]

    # The size of the sensor, and the size of the finger
    SENSOR_SIZE_MM = 38.67
    FINGER_SIZE_MM = 12.5

    GENERATE_INFO_FILE = True

    # Generate the info file:
    if GENERATE_INFO_FILE:
        info = geerate_info_file(VIDEO_FILE, GCODE_FILE, CENTER_POINT, DIR)
    else:
        # Load in the test info and the force data
        info = json.load(open(DIR + 'info.json', 'r'))

    # Parse the wieght data
    force_data:np.ndarray = load_force_data(WIEGHTS_FILE)

    # open the video
    cap = cv2.VideoCapture(DIR + info['video_name'])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use the time stamp from info to zero the time in the force data
    force_data[:, 1] -= time_stamp_to_seconds(info['start_timestamp'])
    for i in range(force_data.shape[0]-1):
        if force_data[i, 1] >= force_data[i+1, 1]:
            print('found issue!')
            print('step', i,":", force_data[i, 1], 'step', i+1, ":", force_data[i+1, 1])

    # Create a spline so that we can convert time to index.
    time_to_frame_spline:UnivariateSpline = scipy.interpolate.UnivariateSpline(force_data[:, 1], np.arange(force_data.shape[0]))

    # find the index of time = 0
    force_start_index:int = int(np.round(time_to_frame_spline(0)))

    # The scale is accurate down to ~1-2 grams (but sometimes can be as far off as 5 grams), so even when no force 
    # is being applied, the scale still reads some force. Inorder to elminate this cases, I set any forces below 
    # 10g to 0g (to represent the sensor is not in contact)
    for i in range(force_data.shape[0]):
        if force_data[i, 0] < 10:
            force_data[i, 0] = 0

    # some may slip through, delete any non-zero measurement whose neighbors and both 0.
    for i in range(1, force_data.shape[0]-1):
        if force_data[i, 0] != 0 and force_data[i-1, 0] == 0 and force_data[i+1, 0] == 0:
            force_data[i, 0] = 0

    # now, split the weight data into individual tests:
    # to do this, first the user selects the peaks of the first two tests.

    print("Please enter the x value of the first peak and the last peak of the test.")
    plt.plot(force_data[force_start_index:, 1], force_data[force_start_index:, 0])
    plt.show()

    while True:
        try:
            first_peak:float = float(input("Enter the first peak time: "))
            break
        except:
            print('invalid input, try again')

    while True:
        try:
            last_peak:float = float(input("Enter the last peak time: "))
            break
        except:
            print('invalid input, try again')

    num_peaks:int = len(info['points'])
    test_length:float = (last_peak - first_peak)/(num_peaks-1)
    peaks:List[float] = [first_peak]
    # The nth peak will be aproximatly at test_length + n-1th peak
    for i in range(1, num_peaks):
        peak_index_min:int = int(np.round(time_to_frame_spline(peaks[-1] + 0.5*test_length)))
        peak_index_max:int = int(np.round(time_to_frame_spline(peaks[-1] + 1.5*test_length)))
        peak_index:int = np.argmax(force_data[peak_index_min:peak_index_max, 0]) + peak_index_min
        peaks.append(force_data[peak_index, 1])

    print("The peaks will now be plotted. Please check to make sure they are correct.")
    # plot the forces, along with vertical lines at the peaks for visual conformation:
    plt.plot(force_data[:, 1], force_data[:, 0])
    for peak in peaks:
        plt.axvline(x=peak, color='r')
    plt.title("Peaks")
    plt.show()

    # Find divisions, the start/ends of each test. divisions are indecies.
    division_times:List[float] = [peaks[0]-test_length/2]
    for i in range(0, len(peaks)-1):
        division_times.append((peaks[i] + peaks[i+1])/2)
    division_times.append(peaks[-1] + test_length/2)

    division_indexs:List[int] = np.round(time_to_frame_spline(division_times)).astype(int).tolist()

    # plot the forces, along with vertical lines at the divisions for visual conformation:
    print("The divisions will now be plotted. Please check to make sure they are correct.")
    plt.plot(force_data[:, 1], force_data[:, 0])
    for division in division_indexs:
        plt.axvline(x=force_data[division, 1], color='r')
    plt.title("Test Divisions")
    plt.show()

    assert len(division_indexs)-1 == len(info['points']), "incorrect number of test divisions"

    # break the video into invidual tests, and save each test video.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # the video is gonna be cropped to square, and the new output size is gonna be:
    defult_num_pixels = 1080
    pixel_size = SENSOR_SIZE_MM/defult_num_pixels

    if os.path.exists(DIR + "processed_data") == False:
        os.mkdir(DIR + "processed_data")

    for i in range(len(info["points"])):
        print('test', i)

        # save each test's video
        start_idx = division_indexs[i]
        end_idx = division_indexs[i+1]
        start_frame = int(force_data[start_idx, 1]*fps) + info["start_frame"]
        end_frame = int(force_data[end_idx, 1]*fps) + info["start_frame"]

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        output = cv2.VideoWriter(DIR + "processed_data/videos" + "test_" + str(i)+ ".mp4", fourcc, fps, (frameWidth, frameHeight))

        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            assert ret, "couldn't read frame"
            output.write(frame)
        output.release()

        # create the force data:
        force_spline = scipy.interpolate.UnivariateSpline(force_data[start_idx:end_idx, 1], force_data[start_idx:end_idx, 0])
        frame_forces = []

        for frame_num in range(start_frame, end_frame):
            frame_time = (frame_num- info["start_frame"])/fps
            pressure = force_spline(frame_time)/(np.pi*(0.5*FINGER_SIZE_MM)**2) # force in g/mm^2
            pressure = pressure*(9.81/1000)*(1000**2)/1000 # g/mm^2 * (N/g) * (m^2/mm^2) * (kN/N) = kN/m^2
            frame_forces.append(pressure)

        location =  np.array(info['points'][i]) - np.array(info['center_point'])
        center_pxl_x = location[0]/pixel_size + defult_num_pixels/2
        center_pxl_y = location[1]/pixel_size + defult_num_pixels/2

        out_dict = {'center_pxl_x': center_pxl_x,
                    'center_pxl_y': center_pxl_y,
                    'contact_radius': FINGER_SIZE_MM/pixel_size/2,
                    'forces':frame_forces,
                    'height':defult_num_pixels,
                    'width':defult_num_pixels}
            
        with open(DIR +  'processed_data/test_' + str(i) + ".pkl", 'wb') as handle:
            pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    cap.release()
