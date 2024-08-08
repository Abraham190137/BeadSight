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
from model_training import plot_pressure_maps

def run_inference(checkpoint_path: str,
                  data_path: str,
                  save_path: str,
                  indicies = None,
                  batch_size: int = 64):
    
    # Load the dataset
    checkpoint = torch.load(checkpoint_path)
    window_size:int = checkpoint['window_size']
    pixel_mean: float = checkpoint['pixel_mean']
    pixel_std: float = checkpoint['pixel_std']
    average_force: float = checkpoint['data_loader_info']['average_force']
    
    model = UNet(window_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if indicies is None: # If no indicies are provided, use all the data
        indicies = get_valid_indices(data_path, window_size)

    dataset = BeadSightDataset(hdf5_file=data_path,
                               indicies=indicies,
                               pixel_mean=pixel_mean,
                               pixel_std=pixel_std,
                               average_force=average_force,
                               train=False,
                               window_size=window_size)
    
    avg_pressure = dataset.avg_pressure
    
    
    test_data_loader = DataLoader(dataset, 
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  num_workers=8, 
                                  prefetch_factor=2, 
                                  pin_memory=True)
    
    with h5py.File(save_path, 'w') as save_file:
        save_file.attrs['checkpoint_path'] = checkpoint_path
        save_file.attrs['data_path'] = data_path
        save_file.attrs['avg_pressure'] = avg_pressure
        
        #copy the checkpoint info the the attributes, excluding the model and optimizer:

        # save_file.attrs['checkpoint_info'] = checkpoint_info

        ground_truth_data = save_file.create_dataset(name='ground_truth', 
                                                shape=(len(indicies), 256, 256),
                                                chunks=(1, 256, 256), 
                                                dtype=np.float32,
                                                compression=9)
        
        predictions_data = save_file.create_dataset(name='predictions', 
                                               shape=(len(indicies), 256, 256),
                                               chunks=(1, 256, 256), 
                                               dtype=np.float32,
                                               compression=9)
        
        data_indicies = save_file.create_dataset(name='indicies',
                                                 shape=(len(indicies),),
                                                 dtype=np.int32)
        
        mse_values = save_file.create_dataset(name='mse_values',
                                                shape=(len(indicies),),
                                                dtype=np.float32)
        
        mae_values = save_file.create_dataset(name='mae_values',
                                                shape=(len(indicies),),
                                                dtype=np.float32)
        
        for i, data in enumerate(tqdm(test_data_loader)):
            video_frames, pressure_frames, idxs = data

            
            video_frames = video_frames.to(device)
            pressure_frames = pressure_frames.to(device)
            
            with torch.no_grad():
                # unnormlize the data
                predictions = model(video_frames)*avg_pressure

                un_norm_pressure_frames = pressure_frames*avg_pressure

                mae_losses = torch.abs(predictions - un_norm_pressure_frames).mean(dim=(1,2)).cpu().numpy()
                mse_losses = ((predictions - un_norm_pressure_frames)**2).mean(dim=(1,2)).cpu().numpy()

                predictions = predictions.cpu().numpy()
                un_norm_pressure_frames = un_norm_pressure_frames.cpu().numpy()


            ground_truth_data[i*batch_size:(i*batch_size + len(idxs))] = un_norm_pressure_frames
            predictions_data[i*batch_size:(i*batch_size + len(idxs))] = predictions
            data_indicies[i*batch_size:(i*batch_size + len(idxs))] = idxs.cpu().numpy()
            mse_values[i*batch_size:(i*batch_size + len(idxs))] = mse_losses[:]
            mae_values[i*batch_size:(i*batch_size + len(idxs))] = mae_losses[:]

def plot_inference_results(data_path: str, 
                           save_folder: str,
                           num_samples: int = 5,
                           indicies: List[int] = None):
    
    os.makedirs(save_folder, exist_ok=True)
    
    with h5py.File(data_path, 'r') as data_file:
        if indicies is None:
            indicies = np.random.choice(len(data_file['predictions']), num_samples)
        
        for idx in indicies:
            ground_truth = data_file['ground_truth'][idx]
            prediction = data_file['predictions'][idx]
            
            save_path = os.path.join(save_folder, f'{data_file['indicies'][idx]}.png')
            plot_pressure_maps(ground_truth, prediction, save_path)
                 


def print_mse_mae(data_path: str):
    with h5py.File(data_path, 'r') as data_file:
        mse_values = data_file['mse_values'][:]
        mae_values = data_file['mae_values'][:]
        avg_pressure = data_file.attrs['avg_pressure']
        print(f'Average Pressure: {avg_pressure}')

        mse = mse_values.mean()
        mae = mae_values.mean()
        
        print(f'MSE: {mse}')
        print(f'MAE: {mae}')

        print(f'MSE Normalized: {mse/(avg_pressure**2)}')
        print(f'MAE Normalized: {mae/avg_pressure}')
        
        

if __name__ == '__main__':
    checkpoint_path = 'data/12_hr_100_0/trained_models/cluster_run_0/checkpoints/checkpoint_71.pt'
    data_path = 'data/12_hr_100_0/hours_11_to_12.hdf5'
    save_folder = 'data/12_hr_100_0/trained_models/cluster_run_0/eval_71/'
    save_name = "11_to_12.hdf5"
    save_path = os.path.join(save_folder, save_name)
    
    run_inference(checkpoint_path, data_path, save_path)
    print_mse_mae(save_path)

    plot_inference_results(save_path, os.path.join(save_folder, "plots"), num_samples=5)

                    