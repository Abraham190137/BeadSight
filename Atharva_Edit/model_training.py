import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import os
import cv2
import time
from tqdm.auto import tqdm
import random, re
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from PIL import Image
from torchvision import transforms
from Unet_model import UNet
import json

class MyDataset(Dataset):
    def __init__(self, video_frames_folder, pressure_maps_folder, transform, mode='train', randomize=False):
        super().__init__()
        self.video_frames_folder = video_frames_folder
        self.pressure_maps_folder = pressure_maps_folder
        self.transform = transform
        self.window_size = 15
        self.mode = mode  # identify mode as 'train' or 'test'

        # # Define the split between training and testing based on test_number
        # self.train_test_split = 80  # Up to test_number 79 for training, the rest for testing

        # Extract unique test_numbers from filenames in video_frames_folder
        test_numbers_set = set()
        for filename in os.listdir(video_frames_folder):
            match = re.match(r'test_(\d+)_frame_\d+\.jpg', filename)
            if match:
                test_number = match.group(1)
                test_numbers_set.add(test_number) 

        # File name to save the test numbers
        json_file_name = "/home/adikshit/Tactile_Sensor/Atharva_Edit/Model_Run_3/test_numbers.json"

        # Convert set to list and shuffle
        if mode == "train" and randomize:
            all_test_numbers = list(test_numbers_set)
            random.shuffle(all_test_numbers)

            # Split the randomized list for training (70%), validation (10%), and testing (20%)
            train_end_idx = int(len(all_test_numbers) * 0.7)
            valid_end_idx = int(len(all_test_numbers) * 0.8)

            # Split the data
            train_test_numbers = all_test_numbers[:train_end_idx]
            valid_test_numbers = all_test_numbers[train_end_idx:valid_end_idx]
            test_test_numbers = all_test_numbers[valid_end_idx:]

            # Saving all the test_numbers in json file:
            # Combine the lists into a dictionary
            test_numbers_dict = {
                "train_test_numbers": train_test_numbers,
                "valid_test_numbers": valid_test_numbers,
                "test_test_numbers": test_test_numbers
            }

            # Write the dictionary to a file in JSON format
            with open(json_file_name, 'w') as file:
                json.dump(test_numbers_dict, file, indent=4)

            print(f'Test numbers saved to {json_file_name}')

        # Load the shuffled test numbers for validation and testing
        else:
            print(f'Reading back the random list from {json_file_name}')
            with open(json_file_name, 'r') as file:
                test_numbers_dict = json.load(file)
                train_test_numbers = test_numbers_dict["train_test_numbers"]
                valid_test_numbers = test_numbers_dict["valid_test_numbers"]
                test_test_numbers = test_numbers_dict["test_test_numbers"]



        # # Split the randomized list into 80% for training and 20% for testing
        # split_index = int(len(all_test_numbers) * 0.8)
        # train_test_numbers = all_test_numbers[:split_index]
        # test_test_numbers = all_test_numbers[split_index:]

        # Initialize lists to hold the paths
        # self.video_frames_paths = []
        self.video_frames=[]
        self.pressure_maps_paths = []

        # Load test_numbers based on mode
        if self.mode == 'train':
            self.test_numbers = train_test_numbers
        elif self.mode == 'valid':
            self.test_numbers = valid_test_numbers
        else:  # mode == 'test'
            self.test_numbers = test_test_numbers

        # Load the data
        for test_number in tqdm(self.test_numbers, desc=f'Loading {self.mode.capitalize()} Data', position=0):

            # Load video frames
            frames = []
            for frame_idx in range(1, 129):  #  Changing 136 to 129 due to data
                video_frame_path = os.path.join(self.video_frames_folder, f'test_{test_number}_frame_{frame_idx}.jpg')
            #     frames.append(video_frame_path)
            # self.video_frames_paths.append(frames)

                video_frame = Image.open(video_frame_path)
                video_frame = self.transform(video_frame)*255
                video_frame = video_frame.to(torch.uint8)
                frames.append(video_frame)
            self.video_frames.append(frames)

            # Load pressure maps
            maps = []
            for frame_idx in range(15, 129):  #  Changing 136 to 129 due to data
                pressure_map_path = os.path.join(self.pressure_maps_folder, f'test_{test_number}_frame_{frame_idx}.npz')
                maps.append(pressure_map_path)
            self.pressure_maps_paths.append(maps)

        print(f'{self.mode.capitalize()} data loading done!')

        # print the total size of the video_frames:
        total_size_bytes = 0
        for frame_list in self.video_frames:
            for frame in frame_list:
                total_size_bytes += frame.nelement() * frame.element_size()

        total_size_mb = total_size_bytes / (1024 ** 2)
        total_items = sum(len(frame_list) for frame_list in self.video_frames)
        print(f"Total size of video frames is {total_size_mb} MB ~ {round(total_size_mb / 1024, 2)} GB. Total number of frames: {total_items}")

        # import sys
        # total_size_bytes = sum(frame.element_size() * frame.nelement() for frame in self.video_frames)
        # total_size_mb = total_size_bytes / (1024 ** 2)  # Convert from bytes to megabytes
        # print(f"Total size of video frames is {total_size_mb} MB for {self.mode.capitalize()} Data")
        # print(f"Shape of vide frame: {np.array(self.video_frame).shape} for {self.mode.capitalize()} Data")
        # print(f"Total size of video frames is {sys.getsizeof(self.video_frames)/(1024**2)} MB")


    def __len__(self):
        return len(self.test_numbers) * (128 + 1 - self.window_size)  #  Changing 135 + 1 to 128 + 1 due to data

    def __getitem__(self, idx):
        test_idx = idx // (128 + 1 - self.window_size)  #  Changing 135 + 1 to 128 + 1 due to data
        frame_idx = idx % (128 + 1 - self.window_size)  #  Changing 135 + 1 to 128 + 1 due to data

        # Load the video frames for the current idx and concatenate them together.
        # video_frames_path = self.video_frames_paths[test_idx][frame_idx]
        # print(f"frame_idx: {frame_idx}, frame_idx + self.window_size: {frame_idx+self.window_size}")
        # print(f"self.video_frames_paths[test_idx][frame_idx]:\n{self.video_frames_paths[test_idx][frame_idx:frame_idx+self.window_size]}")
        # print(f"Shape of self.video_frames_paths[test_idx][frame_idx:frame_idx+self.window_size]: {np.array(self.video_frames_paths[test_idx][frame_idx:frame_idx+self.window_size]).shape}")
        # print(f"Shape of self.video_frames_paths: {np.array(self.video_frames_paths).shape}")
        
        # video_frames = []

        # Start the timer
        # video_start_time = time.time()

        # Load the video frames for the current idx and concatenate them together.

        # for image_path in self.video_frames_paths[test_idx][frame_idx:frame_idx+self.window_size]:
        #     # print(f"image_path: {image_path}")
        #     video_frame = Image.open(image_path)
        #     video_frame = self.transform(video_frame)*255
        #     video_frame = video_frame.to(torch.uint8)
        #     video_frames.append(video_frame)
        video_frames = self.video_frames[test_idx][frame_idx:frame_idx+self.window_size]
        video_frames = torch.cat(video_frames, dim=0).to(torch.float32) / 255

        # End the timer
        # video_end_time = time.time()

        # Calculate the duration
        # video_duration = video_end_time - video_start_time

        # print(f"The video_frame code block took {video_duration} seconds to run.")

        # Load the pressure map for the current idx on-the-fly
         # Start the timer
        # pressure_start_time = time.time()

        pressure_map_path = self.pressure_maps_paths[test_idx][frame_idx]
        pressure_map = np.load(pressure_map_path)['arr_0']
        
        # # End the timer
        # pressure_end_time = time.time()

        # # Calculate the duration
        # pressure_duration = pressure_end_time - pressure_start_time

        # print(f"The video_frame code block took {pressure_duration} seconds to run.")

        # Calculate the average value of the pressure map
        average_pressure = np.mean(pressure_map)

        # Check if the average pressure is greater than 1.5
        if average_pressure > 0:
            # If the average is greater than 1.5, keep the pressure map as it is
            pressure_maps = torch.from_numpy(pressure_map).unsqueeze(0).float() / 1000
        else:
            # If the average is not greater than 5, set the pressure map values to 0
            pressure_map[:] = 0  # This sets all values in the pressure_map to 0
            pressure_maps = torch.from_numpy(pressure_map).unsqueeze(0).float() / 1000

        if self.mode == 'train':
            # Choose a random augmentation: 0 for vertical flip, 1 for horizontal flip, 2 for rotation
            augmentation_choice = random.randint(0, 2)

            if augmentation_choice == 0:
                # Vertical flip
                video_frames = TF.vflip(video_frames)
                pressure_maps = TF.vflip(pressure_maps)

            elif augmentation_choice == 1:
                # Horizontal flip
                video_frames = TF.hflip(video_frames)
                pressure_maps = TF.hflip(pressure_maps)

            else:
                # Random 90-degree rotation
                rotations = [0, 90, 180, 270]
                rotation_angle = random.choice(rotations)
                video_frames = TF.rotate(video_frames, rotation_angle)
                pressure_maps = TF.rotate(pressure_maps, rotation_angle)

        # Retrieve the file name of the frame used for validation
        pressure_frame_file_name = os.path.basename(self.pressure_maps_paths[test_idx][frame_idx])

        return video_frames, pressure_maps, pressure_frame_file_name
    
# Saving pressure maps
def plot_pressure_maps(ground_truth, prediction, save_directory):
    
    # Calculate the min and max values for the colorbar range
    cmin = min(np.min(ground_truth), np.min(prediction))
    cmax = max(np.max(ground_truth), np.max(prediction))

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the ground truth pressure map
    img1 = ax1.imshow(ground_truth, cmap='gray', vmin=cmin, vmax=cmax)
    ax1.set_title('Ground Truth PM')
    ax1.axis('off')  # Removes the axis for a cleaner look

    # Plot the predicted pressure map
    img2 = ax2.imshow(prediction, cmap='gray', vmin=cmin, vmax=cmax)
    ax2.set_title('Predicted PM')
    ax2.axis('off')  # Removes the axis for a cleaner look

    # Add a color bar to the right of the subplots
    fig.colorbar(img1, ax=[ax1, ax2], orientation='vertical')

    # # Save the plot
    # plt.tight_layout()

    # Save the figure
    plt.savefig(save_directory, bbox_inches='tight')

    # Close the figure
    plt.close(fig) 
    


# Model Training:
def model_training (model, train_dataloader, valid_dataloader, criterion, optimizer, device, n_epochs=100, checkpoint_path='best_model.pth'):
    # Initialize the start epoch to 0
    start_epoch = 0
    # Initialize the best loss to a high value
    best_loss = float('inf')

    # Lists to store loss and epoch values
    loss_values = []
    epoch_values = []
    validation_losses = []

    # Check if there's a checkpoint from which to continue training
    if os.path.isfile(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Continue from the next epoch
        best_loss = checkpoint['best_loss']
        loss_values = checkpoint['loss_values']  # Load traning loss values
        validation_losses = checkpoint['validation_losses']  # Load validation loss values
        epoch_values = checkpoint['epoch_values']  # Load epoch values
        print(f"Resumed training from epoch {start_epoch}")

    print("Start training...")

    for epoch in range(start_epoch, n_epochs):

        model.train()  # Set the model to training mode
        running_loss = 0.0

        # Iterate over the dataloader
        for i, (video_frames, pressure_maps, _) in enumerate(tqdm(train_dataloader, desc=f'Training epoch {epoch+1}/{n_epochs}')):
            video_frames = video_frames.to(device)
            pressure_maps = pressure_maps.to(device)

            # Forward pass
            predictions = model(video_frames)  # Pass the video frames through the model
            loss = criterion(predictions, pressure_maps)  # Calculate mse loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # Add the loss to the running loss

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        validation_loss = 0.0
        with torch.no_grad():
            for i, (video_frames, pressure_maps, valid_file_name) in enumerate(tqdm(valid_dataloader, desc=f'Validating')):
                video_frames = video_frames.to(device)
                pressure_maps = pressure_maps.to(device)
                predictions = model(video_frames)
                val_loss = criterion(predictions, pressure_maps)
                validation_loss += val_loss.item() * video_frames.size(0)  # Accumulate the validation loss

                # Visualization of validation batch
                if i % 25 == 0:   # Adjust the frequency of visualization as needed
                    match = re.match(r"(test_\d+)_frame_\d+", valid_file_name[0])
                    if match:
                        valid_name = match.group(1)
        
                    pred = predictions[0, 0, :, :].detach().cpu().numpy()
                    ground_truth = pressure_maps[0, 0, :, :].detach().cpu().numpy()

                    # Define the path where you want to save the image
                    save_path_images = "/home/adikshit/Tactile_Sensor/Atharva_Edit/Model_Run_3/Validation Images"
                    if not os.path.exists(save_path_images):
                        os.makedirs(save_path_images)  # Creates the directory if it does not exist

                    plot_pressure_maps(ground_truth, pred, os.path.join(save_path_images, f"validation_visualization_epoch_{epoch}_batch_{i}_[{valid_name}].png"))

                    # Saving the visualized prediction and ground truth PM as npz files:
                    save_path_data = "/home/adikshit/Tactile_Sensor/Atharva_Edit/Model_Run_3/Validation Data"
                    if not os.path.exists(save_path_data):
                        os.makedirs(save_path_data)  # Creates the directory if it does not exist
                    np.savez_compressed(os.path.join(save_path_data, f"predicted_PM_epoch_{epoch}_batch_{i}_[{valid_name}].npz") , pred)
                    np.savez_compressed(os.path.join(save_path_data, f"groundTruth_PM_epoch_{epoch}_batch_{i}_[{valid_name}].npz") , ground_truth)

        # Calculate the average losses for the epoch

        avg_loss = running_loss / len(train_dataloader)
        avg_val_loss = validation_loss / len(valid_dataloader.dataset)

        # Append values to lists for plotting later
        loss_values.append(avg_loss)
        epoch_values.append(epoch + 1)
        validation_losses.append(avg_val_loss)

        # Check if this is the best model so far
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'loss_values': loss_values,
                'validation_losses': validation_losses,
                'epoch_values': epoch_values
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

        print(f'Epoch {epoch+1}/{n_epochs} | Training Loss: {avg_loss:.5e} | Validation Loss: {avg_val_loss:.5e}')


def model_testing(model, test_dataloader, criterion, device, model_path):
    # Load the saved model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()

    # Initialize empty lists to store losses and predictions
    test_losses = []
    test_avg_losses = []

    total_test_loss = 0.0

    # Iterate over the test dataset
    for i, (video_frames, pressure_maps, test_file_name) in enumerate(tqdm(test_dataloader, desc=f'Testing')):
        video_frames, pressure_maps = video_frames.to(device), pressure_maps.to(device)

        with torch.no_grad():
            # Forward pass
            predictions = model(video_frames)

            # Calculate loss
            loss = criterion(predictions, pressure_maps)
            test_losses.append(loss.item())
            total_test_loss += loss.item() * video_frames.size(0)  # Multiply by batch size to accumulate correctly

        test_avg_losses.append(total_test_loss / len(test_dataloader.dataset))

        # Visualization for the first sample in the batch
        match = re.match(r"(test_\d+)_frame_\d+", test_file_name[0])
        if match:
            test_name = match.group(1)

        pred = predictions[0, 0, :, :].detach().cpu().numpy()
        ground_truth = pressure_maps[0, 0, :, :].detach().cpu().numpy()
        
        # Define the path where you want to save the image
        save_path_images = "/home/adikshit/Tactile_Sensor/Atharva_Edit/Model_Run_3/Testing Images"
        if not os.path.exists(save_path_images):
            os.makedirs(save_path_images)  # Creates the directory if it does not exist

        plot_pressure_maps(ground_truth, pred, os.path.join(save_path_images, f"testing_image_{i}[{test_name}].png"))

        # Saving the visualized prediction and ground truth PM as npz files:
        save_path_data = "/home/adikshit/Tactile_Sensor/Atharva_Edit/Model_Run_3/Testing Data"
        if not os.path.exists(save_path_data):
            os.makedirs(save_path_data)  # Creates the directory if it does not exist
        np.savez_compressed(os.path.join(save_path_data, f"predicted_PM_iter_{i}_[{test_name}].npz") , pred)
        np.savez_compressed(os.path.join(save_path_data, f"groundTruth_PM_iter_{i}_[{test_name}].npz") , ground_truth)

    print(f"\n Updating the checkpoint with testing loss...")
    # Update the checkpoint dictionary with the testing loss
    checkpoint['testing_losses'] = test_avg_losses 

    # Save the updated checkpoint
    torch.save(checkpoint, model_path)


if __name__ == '__main__':
    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),  # Resize the images to 256x256
    #     transforms.ToTensor(),  # Convert the images to PyTorch tensors
    # ])

    # # Check if CUDA is available and set the device to GPU, otherwise use CPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print(f'Using device: {device}')

    # # Create DataLoader
    # # If "test_numbers.json" is not present, run train_dataset ONLY with parameter "randomize=True" [randomize is set to False by default for all]
    # train_dataset = MyDataset('/home/adikshit/old_processed_data/sensor_video_files', '/home/adikshit/old_processed_data/sensor_pressure_files', transform, mode='train')
    # valid_dataset = MyDataset('/home/adikshit/old_processed_data/sensor_video_files', '/home/adikshit/old_processed_data/sensor_pressure_files', transform, mode='valid')
    # test_dataset = MyDataset('/home/adikshit/old_processed_data/sensor_video_files', '/home/adikshit/old_processed_data/sensor_pressure_files', transform, mode='test')

    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, prefetch_factor=6)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=8, prefetch_factor=6)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, prefetch_factor=6)

    # # Define the model, criterion, and optimizer
    # model = UNet().to(device)
    # criterion = torch.nn.MSELoss().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # # Define the model path
    # model_path = '/home/adikshit/Tactile_Sensor/Atharva_Edit/Model_Run_3/best_model.pth'

    # # Train the model
    # # model_training(model, train_dataloader, valid_dataloader, criterion, optimizer, device, n_epochs=100, checkpoint_path=model_path)

    # # Testing the model
    # model_testing(model, test_dataloader, criterion, device, model_path)

    # # Loading values for plotting the training, validation and testing losses:
    # model = torch.load(model_path)
    # training_loss = model['loss_values']  # Load training loss values
    # validation_losses = model['validation_losses']  # Load validation loss values
    # testing_losses = model['testing_losses']  # Load testing loss values
    # epoch_values = model['epoch_values']  # Load epoch values

    # # Plotting the training and validation losses
    # train_val_save_path = os.path.join('/home/adikshit/Tactile_Sensor/Atharva_Edit/Model_Run_3', f"training_and_validation_loss_2.png")
    # print(f"\nPlotting and Saving training-validation losses from the model at {train_val_save_path}")
    # plt.figure()
    # plt.plot(epoch_values[1:20], training_loss[1:20], label='Training Loss')
    # plt.plot(epoch_values[1:20], validation_losses[1:20], label='Validation Loss')
    # plt.title('Training & Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # # Save the figure
    # plt.savefig(train_val_save_path, bbox_inches='tight')
    # plt.close()

    # # Plotting the testing loss
    # test_save_path = os.path.join('/home/adikshit/Tactile_Sensor/Atharva_Edit/Model_Run_3', f"testing_loss.png")
    # print(f"\nPlotting and Saving testing loss from the model at {test_save_path}")
    # plt.figure()
    # plt.plot(testing_losses, label='Testing Loss')
    # plt.title('Testing Loss')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.legend()
    # # Save the figure
    # plt.savefig(test_save_path, bbox_inches='tight')
    # plt.close()

    # # print("Testing Loss:\n")
    # # for i in range(len(testing_losses)):
    # #     print(testing_losses[i])

    ground_truth = "/home/adikshit/Tactile_Sensor/Atharva_Edit/Model_Run_3/Testing Data/groundTruth_PM_iter_70_[test_414].npz"
    pred = "/home/adikshit/Tactile_Sensor/Atharva_Edit/Model_Run_3/Testing Data/predicted_PM_iter_70_[test_414].npz"
    plot_pressure_maps(ground_truth, pred, "/home/adikshit/Tactile_Sensor/Atharva_Edit/Model_Run_3/PM_img_test_414.png")