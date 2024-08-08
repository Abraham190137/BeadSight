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
from PIL import Image
import io

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

def plot_pressure_maps(ground_truth, prediction, save_path, images = None, axis_off=True, image_std=[1, 1, 1], image_mean=[0, 0, 0]):
    """
    Plots the ground truth and predicted pressure maps side by side
    ground_truth: np.array - the ground truth pressure map (HxW) 
    prediction: np.array - the predicted pressure map (HxW)
    images: np.array - the input images (3TxHxW) or (Tx3xHxW) - I've added some flexibility here - the chanels can be in almost any order
    save_path: str - the path to save the plot
    """

    # deal with shapes:

    # make sure the gt and prediction are the same shape and 2d
    assert ground_truth.ndim == 2, "Ground truth pressure map must be 2D"
    assert prediction.ndim == 2, "Predicted pressure map must be 2D"
    assert ground_truth.shape == prediction.shape, "Ground truth and prediction must have the same shape"

    if images is not None:
        # deal with the images
        h = ground_truth.shape[0]
        w = ground_truth.shape[1]

        # reshape the images to be Tx3xHxW
        # first, establish the H and W indices:
        if h == w:
            hw_idxs = np.where(np.array(images.shape) == h)[0]
            assert len(hw_idxs) >= 2, "Images must have two dimensions of the same size"
            hw_idxs = hw_idxs[-2:] # take the last two indices, incase h or w equal 3, T, or 3T

            h_idx = hw_idxs[0]
            w_idx = hw_idxs[1]
        
        else:
            h_idx = np.where(np.array(images.shape) == h)[0][-1] # take the last index, incase h or w equal 3, T, or 3T
            w_idx = np.where(np.array(images.shape) == w)[0][-1]
            
        if images.ndim == 3:
            c_idx = list({0, 1, 2} - {h_idx, w_idx})[0]
            images = np.moveaxis(images, [c_idx, h_idx, w_idx], [0, 1, 2]) 

            # split c_idx into T and C, 
            images = images.reshape(-1, 3, h, w)

            # Make the images plt friendly, by putting them into T x H x W x C
            images = images.transpose(0, 2, 3, 1)

        elif images.ndim == 4:
            ct_idxs = list({0, 1, 2, 3} - {h_idx, w_idx})
            # typically, c comes after t:
            if images.shape[ct_idxs[1]] == 3: # if both c and t are 3, then this assumes c comes after t
                c_idx = ct_idxs[1]
                t_idx = ct_idxs[0]
            elif images.shape[ct_idxs[0]] == 3:
                c_idx = ct_idxs[0]
                t_idx = ct_idxs[1]
            else:
                raise ValueError("Images must have a channel dimension of size 3")
            
            images = np.moveaxis(images, [t_idx, h_idx, w_idx, c_idx], [0, 1, 2, 3])

        else:
            raise ValueError("Images must have either 3 or 4 dimensions")
        
        # lets make sure this worked...
        assert images.shape[1] == h, "Image processing failed, h dimension is incorrect"
        assert images.shape[2] == w, "Image processing failed, w dimension is incorrect"
        assert images.shape[3] == 3, "Image processing failed, c dimension is incorrect"

        # next, we need to unnormalize the images (along the channel axis)
        image_std = np.array(image_std)[None, None, None, :]
        image_mean = np.array(image_mean)[None, None, None, :]
        images = images * image_std + image_mean

        if images.max() > 1: # likely the images are in the range 0-255
            images = images / 255

        images = images.clip(0, 1)  # clip the images to be in the range 0-1

        # Now lets plot the images:
        diff_images = []
        for t in range(images.shape[0]):
            diff_images.append(np.linalg.norm(images[t] - images[0], axis=2))
        diff_images = np.stack(diff_images, axis=0)

        # Calculate the min and max values for the colorbar range
        cmin = min(np.min(ground_truth), np.min(prediction))
        cmax = max(np.max(ground_truth), np.max(prediction))

        save_figs: List[Image.Image] = []
        for t in range(images.shape[0]):
            # Create a subplot with 1 row and 4 columns
            ax: List[plt.Axes]
            fig: plt.Figure
            fig, ax = plt.subplots(1, 4, figsize=(18, 5))

            # Plot the ground truth pressure map
            img2 = ax[2].imshow(ground_truth, cmap='gray', vmin=cmin, vmax=cmax)
            ax[2].set_title('Ground Truth PM')

            # Plot the predicted pressure map
            img3 = ax[3].imshow(prediction, cmap='gray', vmin=cmin, vmax=cmax)
            ax[3].set_title('Predicted PM')

            # Add a color bar to the right of the subplots
            fig.colorbar(img2, ax=[ax[0], ax[1]], orientation='vertical')

            ax[0].imshow(images[t])
            ax[0].set_title('Video obs')
            
            ax[1].imshow(diff_images[t], cmap='gray', vmax=diff_images.max(), vmin=0)
            ax[1].set_title('Difference Image')

            if axis_off:
                for axis in ax:
                    axis.axis('off')  # Removes the axis for a cleaner look

            # plt.tight_layout()

            # we want to create a gif, so we need to save the images
            fig.canvas.draw()
            fig_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            fig_img = fig_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            save_figs.append(Image.fromarray(fig_img))

        # save the images as a gif
        save_figs[0].save(save_path, save_all=True, append_images=save_figs[1:], duration=33, loop=0)

        # get base path for saving the npz file and image (if provided):
        base_path = save_path.split('.')[0]
        # save the images and the pressure maps in a npz file for later use
        np.savez_compressed(base_path + '.npz', ground_truth=ground_truth, prediction=prediction, images=images)
    
    # if no images are provided, just plot the pressure maps
    else:
        # Calculate the min and max values for the colorbar range
        cmin = min(np.min(ground_truth), np.min(prediction))
        cmax = max(np.max(ground_truth), np.max(prediction))

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the ground truth pressure map
        img1 = ax1.imshow(ground_truth, cmap='gray', vmin=cmin, vmax=cmax)
        ax1.set_title('Ground Truth PM')

        # Plot the predicted pressure map
        img2 = ax2.imshow(prediction, cmap='gray', vmin=cmin, vmax=cmax)
        ax2.set_title('Predicted PM')

        if axis_off:
            ax1.axis('off')  # Removes the axis for a cleaner look
            ax2.axis('off')

        # Add a color bar to the right of the subplots
        fig.colorbar(img1, ax=[ax1, ax2], orientation='vertical')

        # Save the figure
        plt.savefig(save_path, bbox_inches='tight')

        # Close the figure
        plt.close(fig) 

        base_path = save_path.split('.')[0]
        np.savez_compressed(base_path + '.npz', ground_truth=ground_truth, prediction=prediction)
    

def plot_loss(train_losses, test_losses, save_path, val_losses=None):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss')
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
          dropout_prob: float,
          epochs: int,
          window_size: int,
          train_test_split: float,
          batch_size: int,
          n_workers: int,
          validation_data_path: str = None,
          load_checkpoint_path: str = None,
          num_plot_samples: int = 5,
          noise_std:float = 0.0):
    
    # get a list of all folders in the save path:
    folders = os.listdir(save_path)

    # find the next run number:
    run_numbers = []
    for folder in folders:
        if name in folder:
            num = folder.split(name)[-1]
            if num[0] == '_':
                num = num[1:] # remove the underscore
            run_numbers.append(int(num))

    run_num = max(run_numbers) + 1 if run_numbers else 0

    save_folder = os.path.join(save_path, f"{name}_{run_num}")
    os.makedirs(save_folder, exist_ok=False)

    # save the source files
    save_src_files(save_folder)

    plot_folder = os.path.join(save_folder, 'plots')
    checkpoint_folder = os.path.join(save_folder, 'checkpoints')
    os.makedirs(plot_folder, exist_ok=False)
    os.makedirs(checkpoint_folder, exist_ok=False)
    
    # get device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # create the model
    model = UNet(window_size=window_size, dropout_prob=dropout_prob)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
        val_losses = []

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
                                  train=False,
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
    

    if validation_data_path is not None:
        validation_data = BeadSightDataset(hdf5_file=validation_data_path,
                                      indicies=get_valid_indices(validation_data_path, window_size),
                                      pixel_mean=pixel_mean,
                                      pixel_std=pixel_std,
                                      average_force=average_force,
                                      train=False,
                                      window_size=window_size,
                                      process_images=False)
        
        valid_data_loader = DataLoader(validation_data, 
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
                                    save_path=os.path.join(epoch_plot_folder, f'train_{i}.gif'),
                                    images=images[0].detach().cpu().numpy(),
                                    image_std=pixel_std,
                                    image_mean=pixel_mean)
        
        train_losses.append(np.mean(epoch_train_losses))

        # test the model
        model.eval()
        epoch_test_losses = []
        epoch_val_losses = []
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
                                    save_path=os.path.join(epoch_plot_folder, f'test_{i}.gif'),
                                    images=images[0].cpu().numpy(),
                                    image_std=pixel_std,
                                    image_mean=pixel_mean)

            test_losses.append(np.mean(epoch_test_losses))
    
            if validation_data_path is not None:
                for i, (images, pressure_maps, idx) in tqdm(enumerate(valid_data_loader), desc=f'Epoch {epoch} - Validation', total=len(valid_data_loader)):
                    images = images.to(device)
                    pressure_maps = pressure_maps.to(device)

                    images = test_data.image_processing(images)

                    outputs = model(images)
                    loss = criterion(outputs, pressure_maps)
                    epoch_val_losses.append(loss.item())

                    # visualize the output
                    if i < num_plot_samples:
                        plot_pressure_maps(ground_truth=pressure_maps[0].cpu().numpy(), 
                                        prediction=outputs[0].cpu().numpy(), 
                                        save_path=os.path.join(epoch_plot_folder, f'val_{i}.gif'),
                                        images=images[0].cpu().numpy(),
                                        image_std=pixel_std,
                                        image_mean=pixel_mean)
            
                val_losses.append(np.mean(epoch_val_losses))

        # print the losses
        

        if validation_data_path is not None:
            plot_loss(train_losses, test_losses, os.path.join(plot_folder, 'loss_plot.png'), val_losses=val_losses)
            print(f'Epoch {epoch} - Train Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}, Val Loss: {val_losses[-1]}')
        else:
            plot_loss(train_losses, test_losses, os.path.join(plot_folder, 'loss_plot.png'))
            print(f'Epoch {epoch} - Train Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}')


        # save the model
        checkpoint_path = os.path.join(checkpoint_folder, f'checkpoint_{epoch}.pt')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            "val_losses": val_losses,
            'pixel_mean': train_data.image_normalize.mean,
            'pixel_std': train_data.image_normalize.std,
            'avg_pressure': train_data.avg_pressure,
            'data_loader_info': train_data.meta_data,
            'train_indices': train_data.indices,
            'test_indices': test_data.indices,
            'dataset_path': train_data.hdf5_file,
            'lr': lr,
            'weight_decay': weight_decay,
            'dropout_prob': dropout_prob,
            'window_size': window_size,
            'batch_size': batch_size,
            'noise_std': noise_std
        }

        torch.save(checkpoint, checkpoint_path)

def main():
    # use the root dir with the relative path to make an absolute path, so it works in any directory
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    DATA_PATH = os.path.join(root_dir, "data/48_hr_100/hours_1_to_3.hdf5")
    VAL_DATA_PATH = os.path.join(root_dir, "data/48_hr_100/hours_43_to_44.hdf5")
    # DECOMPRESSED_DATA_PATH = os.path.join(root_dir, "data/initial_test_34/decompressed_data.hdf5")
    SAVE_PATH = os.path.join(root_dir, "data/48_hr_100/trained_models")

    # if not os.path.exists(DECOMPRESSED_DATA_PATH):
    #     decompress_h5py(DATA_PATH, DECOMPRESSED_DATA_PATH)

    matplotlib.use('Agg')
    train_model(data_path=DATA_PATH,
                        save_path=SAVE_PATH,
                        validation_data_path= VAL_DATA_PATH,
                        name = "hours_1_to_3_no_rotflip",
                        lr=1e-4,
                        weight_decay=1e-4,
                        dropout_prob=0.2,
                        epochs=100,
                        window_size=15,
                        train_test_split=0.8,
                        batch_size=64,
                        n_workers = 16,
                        noise_std=0.05,
                        num_plot_samples = 5)
    
if __name__ == "__main__":
    main()