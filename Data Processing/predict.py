import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import time
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
from Unet_model import UNet

# Load the model
model = UNet()
model.load_state_dict(torch.load('best_model_712_speedup_80_old.pth'))
model.eval()
testing_loss = 0.0
total_samples = 0

# Define the necessary transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to 256x256
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    # transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # Normalize the images
])


def load_images(video_frames_folder, transform, test_number,window_size,frame_idx):
    video_frames = []
    for i in range(window_size):
        frame_number = frame_idx + i
        video_frame_path = os.path.join(video_frames_folder, f'test_{test_number}_frame_{frame_number}_unwrap.jpg')
        video_frame = Image.open(video_frame_path)
        if transform:
            video_frame = transform(video_frame)
        video_frames.append(video_frame)
    video_frames = torch.cat(video_frames, dim=0)/256 
    return video_frames

# Initialize empty lists to store losses and predictions
losses = []
avg_losses = []
accuracies = []

window_size = 15
for test_number in range(80,100):#,2):
    for frame_idx in range(1, 136-window_size,5):
        images = load_images('../test_712/video_frames_predicted', transform, test_number,window_size,frame_idx)

        
        # Make a prediction and measure the inference time

        start_time = time.time()
        with torch.no_grad():
            prediction = model(images.unsqueeze(0))*1000  # Convert to kN/m^2
        end_time = time.time()
        
        if frame_idx == 1: # only print once
            print(f'Inference time: {end_time - start_time}s')

        target_pressure_path = os.path.join('../test_712/pressure_frames_predicted', f'test_{test_number}_frame_{frame_idx+window_size-1}.npy')    
        target_pressure = np.load(target_pressure_path)
        target_pressure = torch.from_numpy(target_pressure).unsqueeze(0).float()
        
        # Calculate the loss
        loss = F.mse_loss(prediction/1000 , target_pressure/1000)
        losses.append(loss.item())
        testing_loss += loss.item()
        total_samples += 1
        avg_loss = testing_loss / total_samples
        avg_losses.append(avg_loss)

        if frame_idx == 1: # only print once
            print(f'Average testing loss: {avg_loss:.6f}')

 
        # Calculate the accuracy (Mean Absolute Percentage Error - MAPE)
        # absolute_percentage_error = torch.abs((prediction - target_pressure) / target_pressure)
        # mape = torch.mean(absolute_percentage_error).item()
        # accuracies.append(1 - mape)


        # prediction = torch.sigmoid(prediction)

        # # Convert tensor to PIL image
        # print(prediction.shape, type(prediction))
        # fig, ax = plt.subplots()
        # img = ax.imshow(prediction[0, 0,:,:], cmap='gray') #, vmin=0, vmax=1500)
        # plt.savefig(f'test_712/pressure_frames_predicted/test_{test_number}_frame_{frame_idx+window_size-1}_prediction.jpg')
        # Create a figure and axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Display the grayscale image
        img1 = ax1.imshow(target_pressure[0,:,:], cmap='gray', vmin=0, vmax=200)
        ax1.set_title('target pressure')

        img2 = ax2.imshow(prediction[0, 0,:,:], cmap='gray', vmin=0, vmax=200)
        ax2.set_title('prediction pressure')

        # Add a color bar to show the values
        cbar = fig.colorbar(img2, ax=[ax1, ax2],shrink=0.8)

        # plt.savefig(f'test_712/pressure_frames_predicted/test_{test_number}_frame_{frame_idx+window_size-1}_comparison_2.png') # for new
        plt.savefig(f'../test_712/pressure_frames_predicted/test_{test_number}_frame_{frame_idx+window_size-1}_comparison_1.png') # for old

        plt.close()
        # plt.show()


print("Predict Done!")
# Calculate average loss and accuracy
avg_loss_total = sum(losses) / len(losses)
# avg_accuracy = sum(accuracies) / len(accuracies)


# Print and plot the results
print(f'Average testing loss: {avg_loss_total:.6f}')
# print(f'Average testing accuracy: {avg_accuracy:.6f}')

# Plot the loss curve
plt.plot(avg_losses)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Loss Curve')
plt.show()

# # Calculate the x-axis values for the plot (iterations)
# iterations = list(range(len(accuracies)))

# # Plot the accuracy curve
# plt.plot(iterations, accuracies)
# plt.xlabel('Iteration')
# plt.ylabel('Accuracy')
# plt.title('Accuracy Curve')
# plt.show()