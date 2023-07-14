import numpy as np
import cv2
import scipy.interpolate
import matplotlib.pyplot as plt
import pickle

WIEGHTS_FILE = "output\out5.csv"
INFO_FILE = "test_info_file.csv"
VIDEO_FILE = "out5_09_09_58.mp4"
SAVE_FOLDER = "processed data/"

PIXEL_SIZE = 13.67/455
FINGER_SIZE_MM = 10

# load in test info:

def load_info(info_file):
    with open(info_file, 'r') as file:
        info_lines = file.read().split('\n')

    calib_frame, start_frame, end_frame = info_lines[0].split(",")
    calib_frame = int(calib_frame)
    start_frame = int(start_frame)
    end_frame = int(start_frame)

    center_point = np.array(info_lines[1].split(",")).astype(float)

    n_tests = int(info_lines[2])

    # get the testing points:
    points = np.empty((n_tests, 2))
    for i in range(n_tests):
        point = np.array(info_lines[i+3].split(",")).astype(float)
        points[i] = point

    info = {"start_frame": start_frame, 
            "calibration_frame": calib_frame,
            "end_frame":end_frame,
            "center_point": center_point,
            "n_tests": n_tests,
            "points": points}
    
    return info

# Load in the test info and the force data
info = load_info(INFO_FILE)
force_data = np.loadtxt(WIEGHTS_FILE, delimiter=",", dtype=float)

# open the video
cap = cv2.VideoCapture(VIDEO_FILE)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

start_time = (info["start_frame"] - info["calibration_frame"])/fps

force_data[:, 1] -= start_time

force_start_index = 0
for i in range(force_data.shape[0]):
    if force_data[i][1] > 0:
        force_start_index = i
        break
 
# Trim data from before the beging of the testing
force_data = force_data[i:, :]

# The scale is accurate down to ~1-2 grams (but sometimes can be as far off as 5 grams), so even when no force 
# is being applied, the scale still reads some force. Inorder to elminate this cases, I set any forces below 
# 6g to 0g (to represent the sensor is not in contact)
for i in range(force_data.shape[0]):
    if force_data[i, 0] < 6:
        force_data[i, 0] = 0

# some may slip through, delete any non-zero measurement whose neighbors and both 0.
for i in range(force_data.shape[0]):
    if i == 0 or i== force_data.shape[0]-1:
        continue # skip the first and last elements

    if force_data[i, 0] != 0 and force_data[i-1, 0] == 0 and force_data[i+1, 0] == 0:
        force_data[i, 0] = 0

# now, split the weight data into individual tests:
test_divisions = []
test_ongoing = False
zeros_start = 0
for i in range(force_data.shape[0]):
    if test_ongoing:
        if force_data[i, 0] == 0:
            # end of test, set zeros_start:
            zeros_start = i
            test_ongoing = False
    else:
        if force_data[i, 0] != 0:
            # start of test. Set the dividing line between tests to be halfway through the zeros.
            test_ongoing = True
            test_divisions.append((zeros_start + i)//2)

# We start the video halfway between presses, so the first division should actually be at 0
test_divisions[0] = 0
test_divisions.append(force_data.shape[0]-1) # stop at last entry

assert len(test_divisions)-1 == info["n_tests"], "incorrect number of test divisions"

# break the video into invidual tests, and save each test video.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# the video is gonna be cropped to square, and the new output size is gonna be:
out_height = frameHeight
out_width = frameHeight 

for i in range (info["n_tests"]):
    print('test', i)

    # save each test's video
    start_idx = test_divisions[i]
    end_idx = test_divisions[i+1]
    start_frame = int(force_data[start_idx, 1]*fps) + info["start_frame"]
    end_frame = int(force_data[end_idx, 1]*fps) + info["start_frame"]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output = cv2.VideoWriter(SAVE_FOLDER + "test_" + str(i)+ ".mp4", fourcc, fps, (frameHeight, frameHeight))

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        frame = frame[:, (frameWidth-frameHeight)//2:((frameWidth-out_width)//2 + out_width), :]
        assert ret, "couldn't read frame"
        output.write(frame)
    output.release()

    # create the force data:
    force_spline = scipy.interpolate.UnivariateSpline(force_data[start_idx:end_idx, 1], force_data[start_idx:end_idx, 0])
    frame_forces = []
    # frame_times = []
    for frame_num in range(start_frame, end_frame):
        frame_time = (frame_num- info["start_frame"])/fps
        force = force_spline(frame_time)/(np.pi*(0.5*FINGER_SIZE_MM/PIXEL_SIZE)**2) # force in g/mm^2
        force = force*(9.81/1000)*(1000**2)/1000 # g/mm^2 * (N/g) * (m^2/mm^2) * (kN/N) = kN/m^2
        frame_forces.append(force_spline(frame_time))
        # frame_times.append(frame_time)
    delta =  info['points'][i] - info['center_point']
    center_pxl_x = delta[0]/PIXEL_SIZE + out_width/2
    center_pxl_y = delta[1]/PIXEL_SIZE + out_height/2

    out_dict = {'center_pxl_x': center_pxl_x,
                'center_pxl_y': center_pxl_y,
                'contact_radius': FINGER_SIZE_MM/PIXEL_SIZE/2,
                'forces':frame_forces,
                'height':out_height,
                'width':out_width}
        
    with open(SAVE_FOLDER + 'test_' + str(i) + ".pkl", 'wb') as handle:
        pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
cap.release()
