import torch
import torch.nn as nn
import os
import cv2
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from PIL import Image
from torchvision import transforms
from Unet_model import UNet

class MyDataset(Dataset):
    def __init__(self, video_frames_folder, pressure_maps_folder, transform):
        super().__init__()
        self.video_frames_folder = video_frames_folder
        self.pressure_maps_folder = pressure_maps_folder
        self.transform = transform
        self.window_size = 15

        test_numbers = set()
        for filename in os.listdir(self.pressure_maps_folder):
            if filename[:5] == 'test_':
                test_number = filename.split('_')[1]
                test_numbers.add(test_number)
        self.test_numbers = list(test_numbers)

        dataload_start_time = time.time()
        # Cache the input data
        self.video_frames = []
        self.pressure_maps = []
        for test_number in self.test_numbers:
            # Load video frames
            frames = []
            for frame_idx in range(1, 136):  
                video_frame_path = os.path.join(self.video_frames_folder, f'test_{test_number}_frame_{frame_idx}_unwrap.jpg')
                video_frame = Image.open(video_frame_path)
                video_frame = self.transform(video_frame)*255
                # print(video_frame)
                video_frame = video_frame.to(torch.uint8)
                # print(video_frame)
                
                frames.append(video_frame)
            self.video_frames.append(frames)

            # Load pressure maps
            maps = []
            for frame_idx in range(15, 136):  
                pressure_map_path = os.path.join(self.pressure_maps_folder, f'test_{test_number}_frame_{frame_idx}.npy')
                pressure_map = np.load(pressure_map_path)
                maps.append(pressure_map)
            self.pressure_maps.append(maps)
        dataload_end_time = time.time()
        print(f'Data loading time: {dataload_end_time - dataload_start_time:.2f}s')  
        # print size of the data
        # print('video_frames', len(self.video_frames))
        # print('pressure_maps', len(self.pressure_maps))
        print('Data loading done!')

    def __len__(self):
        return len(self.test_numbers) * (135 + 1 - self.window_size)

    def __getitem__(self, idx):
        test_idx = idx // (135 + 1 - self.window_size)
        frame_idx = idx % (135 + 1 - self.window_size)

        # print(f'test_idx: {test_idx}, frame_idx: {frame_idx}')
        # print(f'len(self.video_frames): {len(self.video_frames)}')
        # print(f'len(self.video_frames[test_idx]): {len(self.video_frames[test_idx])}')
        video_frames = self.video_frames[test_idx][frame_idx:frame_idx+self.window_size]
        video_frames = torch.cat(video_frames, dim=0).to(torch.float32) / 255
        # print(video_frames.shape)

        # print(f'test_idx: {test_idx}, frame_idx: {frame_idx}')
        # print(f'len(self.pressure_maps): {len(self.pressure_maps)}')
        # print(f'len(self.pressure_maps[test_idx]): {len(self.pressure_maps[test_idx])}')
        pressure_maps = self.pressure_maps[test_idx][frame_idx]
        pressure_maps = torch.from_numpy(pressure_maps).unsqueeze(0).float() / 1000
        # print(pressure_maps.shape)
    

        return video_frames, pressure_maps
    

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to 256x256
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
])

device = torch.device('mps')
print('device using:', device)

# Create DataLoader
dataset = MyDataset('../test_712/video_frames', '../test_712/pressure_frames', transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# Define the loss function and optimizer
model = UNet().to(device)
criterion = torch.nn.MSELoss().to(device)
# model.load_state_dict(torch.load('best_model_712_speedup.pth'))

# Adjust learning rate here
learning_rate = 1e-3  # adjust this value according to your needs
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
n_epochs = 100    

# Initialize the best loss to a high value
best_loss = float('inf')

# Lists to store loss and epoch values
loss_values = []
epoch_values = []

print("start training")

for epoch in range(n_epochs):
    start_time = time.time()
    
    running_loss = 0.0
    for i, (video_frames, pressure_maps) in enumerate(tqdm(dataloader)):
        # print(i)

        video_frames = video_frames.to(device)
        pressure_maps = pressure_maps.to(device)

        # Forward pass
        # print("start forward Pass")
        predictions = model(video_frames)
        
        if i % 200 == 0:    # print every 200 mini-batches
            # print('video_shape', video_frames[0, 0:3, :, :].shape)
            output_image = predictions[0, 0, :, :].detach().cpu().numpy()
            ground_truth_image = pressure_maps[0, 0, :, :].detach().cpu().numpy()
            max_value = max(np.max(output_image), np.max(ground_truth_image))
            input_image = np.moveaxis(video_frames[0, 0:3, :, :].cpu().numpy(), 0, -1)
            cv2.imshow('input_s', input_image)
            cv2.imshow('output_s', output_image/max_value)
            cv2.imshow('ground truth_s', ground_truth_image/max_value)

            cv2.waitKey(1)
    
        # print("start loss")
        loss = criterion(predictions, pressure_maps)  # Calculate mse loss

        # Backward pass and optimization
        # print("start backward")
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        # print("end for loop")

    end_time = time.time()
    epoch_time = end_time - start_time 
    avg_loss = running_loss / len(dataloader)
    
    # Check if this is the best model so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_model_712_speedup_80.pth')

    print(f"Epoch {epoch+1} of {n_epochs} took {epoch_time:.2f}s")
    print(f"Training loss: {avg_loss:.6f}")

    # Append values to lists
    loss_values.append(avg_loss)
    epoch_values.append(epoch + 1)

# Plotting the training loss
plt.plot(epoch_values, loss_values, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
