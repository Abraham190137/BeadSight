import numpy as np
import cv2
import scipy.interpolate
import matplotlib.pyplot as plt
import pickle
import json
from typing import List, Dict, Tuple
from scipy.interpolate import UnivariateSpline

DIR = "C:/Research/Tactile Sensor/Tactile_Sensor/Data Collection/data/7-12_test_1/"
WIEGHTS_FILE = DIR + "weight_readings_7-12_4.csv"

SENSOR_SIZE_MM = 38.67
FINGER_SIZE_MM = 12.5

def time_stamp_to_seconds(time_stamp:str) -> float:
    time_stamp = time_stamp.split(':')
    seconds = float(time_stamp[2])
    seconds += float(time_stamp[1])*60
    seconds += float(time_stamp[0])*3600
    return seconds

# load in test info:

# def load_info(info_file):
#     with open(info_file, 'r') as file:
#         info_lines = file.read().split('\n')

#     calib_frame, start_frame, end_frame = info_lines[0].split(",")
#     calib_frame = int(calib_frame)
#     start_frame = int(start_frame)
#     end_frame = int(start_frame)

#     center_point = np.array(info_lines[1].split(",")).astype(float)

#     n_tests = int(info_lines[2])

#     # get the testing points:
#     points = np.empty((n_tests, 2))
#     for i in range(n_tests):
#         point = np.array(info_lines[i+3].split(",")).astype(float)
#         points[i] = point

#     info = {"start_frame": start_frame, 
#             "calibration_frame": calib_frame,
#             "end_frame":end_frame,
#             "center_point": center_point,
#             "n_tests": n_tests,
#             "points": points}
    
#     return info

# Load in the test info and the force data
info = json.load(open(DIR + 'info.json', 'r'))

force_data_txt: str = open(WIEGHTS_FILE, 'r').read().split('\n')
force_data: List[List[float]] = []
for line in force_data_txt:
    split_line: List[str] = line.split(',')
    if len(split_line) != 3:
        continue
    force: float = float(split_line[0])
    seconds: float = time_stamp_to_seconds(split_line[2])
    force_data.append((force, seconds))

force_data:np.ndarray = np.array(force_data)

# open the video
cap = cv2.VideoCapture(DIR + info['video_name'])
fps = int(cap.get(cv2.CAP_PROP_FPS))
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# trim the force data to start at the start of the video
start_time:float = time_stamp_to_seconds(info['start_timestamp'])
force_data[:, 1] -= start_time

# Create a spline so that we can convert time to index.
time_to_frame_spline:UnivariateSpline = scipy.interpolate.UnivariateSpline(force_data[:, 1], np.arange(force_data.shape[0]))

# find the index of time = 0
force_start_index = 0
for i in range(force_data.shape[0]):
    if force_data[i][1] > 0:
        force_start_index = i
        break

# The scale is accurate down to ~1-2 grams (but sometimes can be as far off as 5 grams), so even when no force 
# is being applied, the scale still reads some force. Inorder to elminate this cases, I set any forces below 
# 10g to 0g (to represent the sensor is not in contact)
for i in range(force_data.shape[0]):
    if force_data[i, 0] < 10:
        force_data[i, 0] = 0

# some may slip through, delete any non-zero measurement whose neighbors and both 0.
for i in range(force_data.shape[0]):
    if i == 0 or i== force_data.shape[0]-1:
        continue # skip the first and last elements

    if force_data[i, 0] != 0 and force_data[i-1, 0] == 0 and force_data[i+1, 0] == 0:
        force_data[i, 0] = 0

# now, split the weight data into individual tests:
# to do this, first the user selects the peaks of the first two tests.
# plt.plot(force_data[force_start_index:, 1], force_data[force_start_index:, 0])
# plt.show()

# first_peak:float = float(input("Enter the first peak time: "))
# second_peak:float = float(input("Enter the second peak time: "))
first_peak:float = 5.5
second_peak:float = 12.3

test_length:float = (second_peak - first_peak)
num_peaks:int = len(info['points'])
peaks:List[float] = [first_peak, second_peak]
# The nth peak will be aproximatly at test_length + n-1th peak
for i in range(2, num_peaks):
    peak_index_min:int = int(np.round(time_to_frame_spline(peaks[-1] + 0.75*test_length)))
    print('index_min:', peak_index_min)
    peak_index_max:int = int(np.round(time_to_frame_spline(peaks[-1] + 1.25*test_length)))
    print('index_max:', peak_index_max)
    peak_index:int = np.argmax(force_data[peak_index_min:peak_index_max, 0]) + peak_index_min
    print('peak_index:', peak_index)
    peaks.append(force_data[peak_index, 1])
    print('peak:', peaks[-1])

print(peaks)

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
plt.plot(force_data[:, 1], force_data[:, 0])
for division in division_indexs:
    plt.axvline(x=force_data[division, 1], color='r')
plt.title("Test Divisions")
plt.show()

# test_divisions = []
# test_ongoing = False
# zeros_start = 0
# for i in range(force_data.shape[0]):
#     if test_ongoing:
#         if force_data[i, 0] == 0:
#             # end of test, set zeros_start:
#             zeros_start = i
#             test_ongoing = False
#     else:
#         if force_data[i, 0] != 0:
#             # start of test. Set the dividing line between tests to be halfway through the zeros.
#             test_ongoing = True
#             test_divisions.append((zeros_start + i)//2)

# We start the video halfway between presses, so the first division should actually be at 0
# test_divisions[0] = 0
# test_divisions.append(force_data.shape[0]-1) # stop at last entry

assert len(division_indexs)-1 == len(info['points']), "incorrect number of test divisions"

# break the video into invidual tests, and save each test video.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# the video is gonna be cropped to square, and the new output size is gonna be:
defult_num_pixels = 1080
pixel_size = SENSOR_SIZE_MM/defult_num_pixels

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
    # frame_times = []
    for frame_num in range(start_frame, end_frame):
        frame_time = (frame_num- info["start_frame"])/fps
        pressure = force_spline(frame_time)/(np.pi*(0.5*FINGER_SIZE_MM)**2) # force in g/mm^2
        pressure = pressure*(9.81/1000)*(1000**2)/1000 # g/mm^2 * (N/g) * (m^2/mm^2) * (kN/N) = kN/m^2
        frame_forces.append(pressure)
        # frame_times.append(frame_time)
    delta =  np.array(info['points'][i]) - np.array(info['center_point'])
    center_pxl_x = delta[0]/pixel_size + defult_num_pixels/2
    center_pxl_y = delta[1]/pixel_size + defult_num_pixels/2

    out_dict = {'center_pxl_x': center_pxl_x,
                'center_pxl_y': center_pxl_y,
                'contact_radius': FINGER_SIZE_MM/pixel_size/2,
                'forces':frame_forces,
                'height':defult_num_pixels,
                'width':defult_num_pixels}
        
    with open(DIR +  'processed_data/test_' + str(i) + ".pkl", 'wb') as handle:
        pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
cap.release()
