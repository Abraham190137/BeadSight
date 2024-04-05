import cv2
import os
import glob
import numpy as np
import pickle
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import unpack_force_data
from defisheye import Defisheye 


vkwargs = {"fov": 180,
            "pfov": 120,
            "xcenter": None,
            "ycenter": None,
            "radius": None,
            "angle": 0,
            "dtype": "linear",
            "format": "fullframe"
            }

input_folder = "/home/adikshit/old_processed_data/"  # Input folder for processing

# Destination location after processing
video_frame_output_folder = "/home/adikshit/old_processed_data/sensor_video_files/"
pressure_frame_output_folder = "/home/adikshit/old_processed_data/sensor_pressure_files/"

os.makedirs(video_frame_output_folder, exist_ok=True)
os.makedirs(pressure_frame_output_folder, exist_ok=True)

video_files = glob.glob(input_folder + "*.mp4")
pressure_files = glob.glob(input_folder + "*.pkl")
frame_interval = 30  # frame interval of 1 second at 30fps

print("Start Video Input!")

for video_file in tqdm(video_files, desc="Processing Videos"):
    video_name = os.path.basename(video_file).split('.')[0].split('_')[-1]
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Assuming you want to start at 0.5 seconds and end at 5 seconds
    start_frame = int(0.5 * fps)
    end_frame = int(4.8 * fps)  # The shortes video in the current data is 4.8 seconds
    frame_interval = int(fps / 30)  # Assuming 30 frames per video

    # Set the frame to start capturing
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_index = 1

    for frame_count in range(start_frame, end_frame, frame_interval):  # Every video will have 130 frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index == 1:
            obj = Defisheye(frame, **vkwargs)
            x, y, i, j = obj.calculate_conversions()

        unwrapped_img = obj.unwarp(frame)
        output_file = os.path.join(video_frame_output_folder, f'test_{video_name}_frame_{frame_index}.jpg')
        cv2.imwrite(output_file, unwrapped_img)
        frame_index += 1

    # print(f"Video {video_name} has {frame_index} frames")

    cap.release()

print("Video Input Done!")

print("Start Pressure Input!")

max_force = 200

for pressure_file in tqdm(pressure_files, desc="Processing Pressure Data"):
    test_name = os.path.basename(pressure_file).split('.')[0].split('_')[-1]
    force_data = unpack_force_data.unpack_data(pressure_file, resolution=[256, 256])

    # print(f"Force Data: {force_data.shape}")
    ''' Shortest video -> 4.8 seconds : 144 frames  (total count should be 130 pressure maps per test)
        The original data had 151 frames for shortest video. Current data has 144. Thus there will be a net
        151 - 144 = 7 frames difference.
    ''' 
    for i in range(30, 144):  
        frame = force_data[i, :, :]
        # plt.imshow(frame)
        # plt.show()
        output_file_npz = os.path.join(pressure_frame_output_folder, f'test_{test_name}_frame_{i-15}.npz')  # Saved index: 15->130
        # print(output_file_npz)
        np.savez_compressed(output_file_npz, frame)

print("Pressure Input Done!")