import torch
import numpy as np
from torch.utils.data import DataLoader
import h5py
from torchvision import transforms

from typing import List, Tuple, Dict

from Unet_model import UNet
import matplotlib
from matplotlib import pyplot as plt
import os
import shutil

from tqdm import tqdm

from data_loader import BeadSightDataset, get_press_indicies, get_valid_indices, decompress_h5py

SRC_FILES = ["model_training.py", "Unet_model.py", "data_loader.py"]

def save_src_files(save_folder: str):
    """
    Saves a copy of the source files used for training in the save folder
    """
    os.makedirs(save_folder, exist_ok=True)
    base_dir = os.path.dirname(os.path.realpath(__file__))
    for src_file in SRC_FILES:
        src_path = os.path.join(base_dir, src_file)
        dst_path = os.path.join(save_folder, src_file)
        shutil.copy(src_path, dst_path)

def plot_pressure_maps(ground_truth, prediction, save_path):
    
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
    plt.savefig(save_path, bbox_inches='tight')

    # also save the raw data to recreate the plot if needed:
    base_path = save_path.split('.')[0]
    np.savez_compressed(base_path + '.npz', ground_truth=ground_truth, prediction=prediction)

    # Close the figure
    plt.close(fig) 

def plot_loss(train_losses, test_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def train_model(data_path: str, 
          save_path: str,
          name: str,
          lr: float,
          weight_decay: float,
          epochs: int,
          window_size: int,
          train_test_split: float,
          batch_size: int,
          n_workers: int,
          load_checkpoint_path: str = None,
          num_plot_samples: int = 5,
          noise_std:float = 0.0) -> UNet:
    
    # get a list of all folders in the save path:
    folders = os.listdir(save_path)

    # find the next run number:
    run_numbers = []
    for folder in folders:
        if name in folder:
            num = int(folder.split(name)[-1])
            run_numbers.append(num)

    run_num = max(run_numbers) + 1 if run_numbers else 0

    save_folder = os.path.join(save_path, f"{name}{run_num}")
    os.makedirs(save_folder, exist_ok=False)

    # save the source files
    save_src_files(save_folder)

    plot_folder = os.path.join(save_folder, 'plots')
    checkpoint_folder = os.path.join(save_folder, 'checkpoints')
    os.makedirs(plot_folder, exist_ok=False)
    os.makedirs(checkpoint_folder, exist_ok=False)
    
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the model
    model = UNet(window_size=window_size)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss().to(device)
    
    if load_checkpoint_path is not None:
        checkpoint = torch.load(load_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        train_indices = checkpoint['train_indices']
        test_indices = checkpoint['test_indices']
        assert data_path == checkpoint['dataset_path'], "The dataset path in the checkpoint does not match the provided data path"
    else:
        start_epoch = 0
        train_losses = []
        test_losses = []

        # random train test split
        # we want to split the data by press, so we need to get the press indicies
        press_indicies = get_press_indicies(data_path, window_size)
        press_nums = np.arange(len(press_indicies))
        np.random.shuffle(press_nums)
        split_idx = int(train_test_split * len(press_nums))
        train_press_nums = press_nums[:split_idx]
        test_press_nums = press_nums[split_idx:]

        print("Number of Presses:", len(press_indicies))
        
        # each press takes ~10 seconds, which is 300 frames
        valid_indices = get_valid_indices(data_path, window_size)
        print("Expected of Presses:", (max(valid_indices) - min(valid_indices))/300)

        train_indices = []
        for press_num in train_press_nums:
            train_indices.extend(press_indicies[press_num])
        
        test_indices = []
        for press_num in test_press_nums:
            test_indices.extend(press_indicies[press_num])

    # load the data:    
    with h5py.File(data_path, 'r') as data:
        pixel_mean = data.attrs['pixel_mean']
        pixel_std = data.attrs['pixel_std']
        forces = data["forces"][:]
        average_force = np.mean(forces[valid_indices])


    # create the data loaders:
    train_data = BeadSightDataset(hdf5_file=data_path,
                                  indicies=train_indices,
                                  pixel_mean=pixel_mean,
                                  pixel_std=pixel_std,
                                  average_force=average_force,
                                  train=True,
                                  window_size=window_size,
                                  image_noise_std=noise_std,
                                  process_images=False)
    
    test_data = BeadSightDataset(hdf5_file=data_path,
                                 indicies=test_indices,
                                 pixel_mean=pixel_mean,
                                 pixel_std=pixel_std,
                                 average_force=average_force,
                                 train=False,
                                 window_size=window_size,
                                 process_images=False)
    
    # create the data loaders
    train_data_loader = DataLoader(train_data, 
                                   batch_size=batch_size, 
                                   shuffle=True, 
                                   num_workers=n_workers, 
                                   prefetch_factor=2, 
                                   pin_memory=True)
    
    test_data_loader = DataLoader(test_data, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=n_workers, 
                                  prefetch_factor=2, 
                                  pin_memory=True)

    for epoch in range(start_epoch, epochs):
        # create a folder to save the plots for this epoch
        epoch_plot_folder = os.path.join(plot_folder, f'epoch_{epoch}')
        os.makedirs(epoch_plot_folder, exist_ok=False)

        # train the model
        model.train()
        epoch_train_losses = []
        for i, (images, pressure_maps, idx) in tqdm(enumerate(train_data_loader), desc=f'Epoch {epoch} - Training', total=len(train_data_loader)):
            images = images.to(device)
            pressure_maps = pressure_maps.to(device)

            images = train_data.image_processing(images)

            outputs = model(images)
            loss = criterion(outputs, pressure_maps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

            if i < num_plot_samples:
                plot_pressure_maps(ground_truth=pressure_maps[0].detach().cpu().numpy(), 
                                    prediction=outputs[0].detach().cpu().numpy(), 
                                    save_path=os.path.join(epoch_plot_folder, f'train_{i}.png'))
        
        train_losses.append(np.mean(epoch_train_losses))

        # test the model
        model.eval()
        epoch_test_losses = []
        with torch.no_grad():
            for i, (images, pressure_maps, idx) in tqdm(enumerate(test_data_loader), desc=f'Epoch {epoch} - Testing', total=len(test_data_loader)):
                images = images.to(device)
                pressure_maps = pressure_maps.to(device)

                images = test_data.image_processing(images)

                outputs = model(images)
                loss = criterion(outputs, pressure_maps)
                epoch_test_losses.append(loss.item())

                # visualize the output
                if i < num_plot_samples:
                    plot_pressure_maps(ground_truth=pressure_maps[0].cpu().numpy(), 
                                    prediction=outputs[0].cpu().numpy(), 
                                    save_path=os.path.join(epoch_plot_folder, f'test_{i}.png'))

        test_losses.append(np.mean(epoch_test_losses))

        # print the losses
        print(f'Epoch {epoch} - Train Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}')

        # plot the losses
        plot_loss(train_losses, test_losses, os.path.join(plot_folder, 'loss_plot.png'))

        # save the model
        checkpoint_path = os.path.join(checkpoint_folder, f'checkpoint_{epoch}.pt')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'pixel_mean': train_data.image_normalize.mean,
            'pixel_std': train_data.image_normalize.std,
            'avg_pressure': train_data.avg_pressure,
            'data_loader_info': train_data.meta_data,
            'train_indices': train_data.indices,
            'test_indices': test_data.indices,
            'dataset_path': train_data.hdf5_file,
            'lr': lr,
            'weight_decay': weight_decay,
            'window_size': window_size,
            'batch_size': batch_size,
            'noise_std': noise_std
        }

        torch.save(checkpoint, checkpoint_path)

    return model

def main():
    # use the root dir with the relative path to make an absolute path, so it works in any directory
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    DATA_PATH = os.path.join(root_dir, "data/12_hr_100_0/hours_1_to_3.hdf5")
    # DECOMPRESSED_DATA_PATH = os.path.join(root_dir, "data/initial_test_34/decompressed_data.hdf5")
    SAVE_PATH = os.path.join(root_dir, "data/12_hr_100_0/trained_models")

    # if not os.path.exists(DECOMPRESSED_DATA_PATH):
    #     decompress_h5py(DATA_PATH, DECOMPRESSED_DATA_PATH)

    matplotlib.use('Agg')
    model = train_model(data_path=DATA_PATH,
                        save_path=SAVE_PATH,
                        name = "run_",
                        lr=1e-4,
                        weight_decay=1e-5,
                        epochs=100,
                        window_size=15,
                        train_test_split=0.8,
                        batch_size=64,
                        n_workers = 12,
                        noise_std=0.05)
    
if __name__ == "__main__":
    main()