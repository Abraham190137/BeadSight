import torch
import torch.nn as nn
import os
import time
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from PIL import Image
from torchvision import transforms
from Unet_model import UNet

class MyDataset(Dataset):
    def __init__(self, video_frames_folder, pressure_maps_folder, transform=None):
        super().__init__()
        self.video_frames_folder = video_frames_folder
        self.pressure_maps_folder = pressure_maps_folder
        self.transform = transform

        # Assuming that the pressure map and video frames for each test exist and match each other
        self.test_numbers = [filename.split('_')[1] for filename in os.listdir(self.pressure_maps_folder)]
    
    def __len__(self):
        return len(self.test_numbers)
 
    def __getitem__(self, idx):
        test_number = self.test_numbers[idx]

        # Load video frames
        video_frames = []
        for i in range(5):
            video_frame_path = os.path.join(self.video_frames_folder, f'test_{test_number}_frame_{i}.png')
            video_frame = Image.open(video_frame_path)
            if self.transform:
                video_frame = self.transform(video_frame)
            video_frames.append(video_frame)
        video_frames = torch.cat(video_frames, dim=0)  # Concatenate along the channel dimension

        # Load corresponding pressure map
        pressure_map_path = os.path.join(self.pressure_maps_folder, f'test_{test_number}_frame_4.png')
        pressure_map = Image.open(pressure_map_path)
        if self.transform:
            pressure_map = self.transform(pressure_map)
        
        return video_frames, pressure_map


transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the images to 256x256
        transforms.ToTensor(),  # Convert the images to PyTorch tensors
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   Normalize the images
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

    ])

# Create DataLoader
dataset = MyDataset('frame/frame_video', 'frame/frame_pressure', transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# Define the loss function and optimizer
model = UNet()
criterion = torch.nn.BCEWithLogitsLoss()

# Adjust learning rate here
learning_rate = 0.0005  # adjust this value according to your needs
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
# Training loop
n_epochs = 100  

# Initialize the best loss to a high value
best_loss = float('inf')

for epoch in range(n_epochs):
    start_time = time.time()
    
    running_loss = 0.0
    for i, (video_frames, pressure_maps) in enumerate(dataloader):
        # Forward pass
        predictions = model(video_frames)
        loss = criterion(predictions, pressure_maps)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    end_time = time.time()
    epoch_time = end_time - start_time
    avg_loss = running_loss / len(dataloader)
    
    # Check if this is the best model so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_model.pth')

    print(f"Epoch {epoch+1} of {n_epochs} took {epoch_time:.2f}s")
    print(f"Training loss: {avg_loss:.4f}")
