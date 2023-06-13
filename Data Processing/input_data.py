import cv2
import os
import glob
import numpy as np
import unpack_force_data


# input_folder = "test/"
input_folder = "predict/"
output_folder = "frame/"
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

video_files = glob.glob(input_folder + "*.mp4")
input_files = glob.glob(input_folder + "*.pkl")
frame_interval = 30  # Save a frame every second

# Process video inputs
# Save the video frames in folder frame_video
# For each video only save 5 figures (from 3.5 seconds to 4 seconds, which is 3.6s, 3.7s, 3.8s, 3.9s, 4.0s)
# The name should be test_1_frame_0, test_1_frame_1, ..., test_1_frame_5, test_2_frame_0, ..., test_59_frame_4


# video_frame_output_folder = os.path.join(output_folder, "frame_video")
video_frame_output_folder = os.path.join(output_folder, "predict_video")

os.makedirs(video_frame_output_folder, exist_ok=True)

for video_file in video_files:
    video_name = os.path.splitext(os.path.basename(video_file))[0].split('_')[-1]  # extract the test number

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_count > frame_interval * 3.5) and (frame_count <= frame_interval * 4) and ((frame_count - frame_interval * 3.5) % 3 == 0):
            output_file = os.path.join(video_frame_output_folder, f'test_{video_name}_frame_{frame_index}.png')
            cv2.imwrite(output_file, frame)
            frame_index += 1

        frame_count += 1

    cap.release()


# Process force data inputs
# Save the pressure frames in folder frame_pressure
# For each test only save the fifth figure (frame on the 4.0 second)
# The name should be test_1_frame_4, test_2_frame_4, ..., test_30_frame_4

# pressure_frame_output_folder = os.path.join(output_folder, "frame_pressure")
pressure_frame_output_folder = os.path.join(output_folder, "predict_pressure")

os.makedirs(pressure_frame_output_folder, exist_ok=True)

for input_file in input_files:
    input_name = os.path.splitext(os.path.basename(input_file))[0].split('_')[-1]  # extract the test number

    force_data = unpack_force_data.unpack_data(input_file)
    normalized_data = (255 * force_data / np.max(force_data)).astype(np.uint8)

    frame = cv2.cvtColor(normalized_data[4 * frame_interval, :, :], cv2.COLOR_GRAY2BGR)
    output_file = os.path.join(pressure_frame_output_folder, f'test_{input_name}_frame_4.png')
    cv2.imwrite(output_file, frame)

print("Input Done!")