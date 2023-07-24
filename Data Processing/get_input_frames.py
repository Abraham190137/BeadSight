import cv2
import os
import glob
import numpy as np
import unpack_force_data
import matplotlib.pyplot as plt

input_folder = "../test_713/"

video_files = glob.glob(input_folder + "processed_data/*.mp4")
pressure_files = glob.glob(input_folder + "processed_data/*.pkl")
frame_interval = 30  # Save a frame every second

# Process video inputs
video_frame_output_folder = os.path.join(input_folder, "video_frames")
os.makedirs(video_frame_output_folder, exist_ok=True)

print("Start Video Input!")

for video_file in video_files:
    video_name = os.path.splitext(os.path.basename(video_file))[0].split('_')[-1]  # extract the test number

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    frame_index = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate the current time in seconds
        current_time = frame_count / fps

        # Set the time range from 0.5 seconds to 5 seconds     # shortest video has 151 frames
        if current_time > 0.5 and current_time <= 5:
            output_file = os.path.join(video_frame_output_folder, f'test_{video_name}_frame_{frame_index}.jpg')
            # resize the frame to 256x256
            # frame = cv2.resize(frame, (256, 256))
            cv2.imwrite(output_file, frame)
            frame_index += 1

        frame_count += 1

    cap.release()

print("Video Input Done!")

# Process force data inputs
pressure_frame_output_folder = os.path.join(input_folder, "pressure_frames")
os.makedirs(pressure_frame_output_folder, exist_ok=True)

# Find the maximum force value in all the files
max_force = 200

print("Start Pressure Input!")

for pressure_file in pressure_files:
    input_name = os.path.splitext(os.path.basename(pressure_file))[0].split('_')[-1]  # extract the test number

    force_data = unpack_force_data.unpack_data(pressure_file, resolution=[256, 256])


    for i in range(30, 151):
        frame = force_data[i, :, :]

        # print(frame.shape)
        # print(frame)

        # Save the frame using np.save
        output_file = os.path.join(pressure_frame_output_folder, f'test_{input_name}_frame_{i-15}.npy')
        np.save(output_file, frame)


        # Create a figure and axes
        fig, ax = plt.subplots()


        # Display the image with the custom colormap
        img = ax.imshow(frame, cmap="gray", vmin=0, vmax=max_force)

        # Resize the output image to 256x256
        output_image = img.get_array()
        output_image_resized = cv2.resize(output_image, (256, 256))

        # Update the image data with the resized version
        img.set_data(output_image_resized)

        # Add a color bar to show the density
        cbar = fig.colorbar(img, ax=ax)

        # Save the figure
        output_file = os.path.join(pressure_frame_output_folder, f'test_{input_name}_frame_{i-15}.png')
        plt.savefig(output_file)

        # Close the figure
        plt.close(fig)

print("Pressure Input Done!")
