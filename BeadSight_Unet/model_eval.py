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
    average_force: float = checkpoint['average_force']
    
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
        
        # copy the checkpoint info the the attributes, excluding the model and optimizer:
        checkpoint_info = {}
        for key, value in checkpoint.items():
            if key not in ['model_state_dict', 'optimizer_state_dict']:
                checkpoint_info[key] = value
        save_file.attrs['checkpoint_info'] = checkpoint_info

        ground_truth = save_file.create_dataset(name='ground_truth', 
                                                shape=(len(indicies), 256, 256),
                                                chunks=(1, 256, 256), 
                                                dtype=np.float32,
                                                compression=9)
        
        predictions = save_file.create_dataset(name='predictions', 
                                               shape=(len(indicies), 256, 256),
                                               chunks=(1, 256, 256), 
                                               dtype=np.float32,
                                               compression=9)
        
        data_indicies = save_file.create_dataset(name='indicies',
                                                 shape=(len(indicies),),
                                                 dtype=np.int32)
        
        for i, data in enumerate(tqdm(test_data_loader)):
            video_frames, pressure_frames, idxs = data

            un_norm_pressure_frames = (pressure_frames*avg_pressure).cpu().numpy()
            
            video_frames = video_frames.to(device)
            pressure_frames = pressure_frames.to(device)
            
            with torch.no_grad():
                # unnormlize the data
                predictions = model(video_frames)*avg_pressure
                predictions = predictions.cpu().numpy()

            ground_truth[i*batch_size:i*batch_size + len(idxs)] = un_norm_pressure_frames
            predictions[i*batch_size:i*batch_size + len(idxs)] = predictions
            data_indicies[i*batch_size:i*batch_size + len(idxs)] = idxs.cpu().numpy()

if __name__ == '__main__':
    checkpoint_path = 'path/to/checkpoint.pth'
    data_path = 'path/to/data.hdf5'
    save_path = 'path/to/save.hdf5'
    
    run_inference(checkpoint_path, data_path, save_path)
                    