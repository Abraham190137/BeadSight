import numpy as np
import cv2
import unpack_force_data
import matplotlib.pyplot as plt

for i in range(100):
    print(i)
    DIR = "C:/Research/Tactile Sensor/Tactile_Sensor/Data Collection/data/7-12_test_1/"
    TEST_NUM = i
    PKL_FILE = DIR + "processed_data/test_" + str(TEST_NUM) + ".pkl"
    SAVE_FILE = DIR + "Demo_Force_Videos/test_" + str(TEST_NUM) + ".mp4"

    # create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    # can set a coustom size
    size = [1080, 1080]
    output = cv2.VideoWriter(SAVE_FILE, fourcc, fps, (size[1], size[0]))

    # pass the path to the pickle file you want to analyze, along with the new size,
    # if you don't wanna use the default size of 1080x1080
    if size != [1080, 1080]:
        force_data = unpack_force_data.unpack_data(PKL_FILE, size)
    else:
        force_data = unpack_force_data.unpack_data(PKL_FILE)

    normalized_data = (255*force_data/np.max(force_data)).astype(np.uint8)

    for i in range(force_data.shape[0]):
        frame = cv2.cvtColor(normalized_data[i, :, :], cv2.COLOR_GRAY2BGR)
        output.write(frame)

    output.release()