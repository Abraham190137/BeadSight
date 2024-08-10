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
import cv2

from tqdm import tqdm

from data_loader import BeadSightDataset, get_press_indicies, get_valid_indices, decompress_h5py
from model_training import plot_pressure_maps

from multiprocessing import Pool


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

        indicies = save_file.create_dataset(name='indicies',
                                                shape=(len(indicies),),
                                                dtype=np.int32)


        gt_pressure_map = save_file.create_dataset(name='gt_pressure_map', 
                                                shape=(len(indicies), 256, 256),
                                                chunks=(1, 256, 256), 
                                                dtype=np.float32,
                                                compression=9)
        
        pred_pressure_map = save_file.create_dataset(name='pred_pressure_map', 
                                               shape=(len(indicies), 256, 256),
                                               chunks=(1, 256, 256), 
                                               dtype=np.float32,
                                               compression=9)
        
        gt_center_of_pressure = save_file.create_dataset(name='gt_center_of_pressure',
                                                         shape=(len(indicies), 2),
                                                         dtype=np.float32)
        
        pred_center_of_pressure = save_file.create_dataset(name='pred_center_of_pressure',
                                                           shape=(len(indicies), 2),
                                                           dtype=np.float32)
        
        gt_total_force = save_file.create_dataset(name='gt_total_force',
                                                    shape=(len(indicies),),
                                                    dtype=np.float32)
        
        pred_total_force = save_file.create_dataset(name='pred_total_force',
                                                    shape=(len(indicies),),
                                                    dtype=np.float32)
        
        otzu_intersection = save_file.create_dataset(name='otzu_intersection',
                                                shape = (len(indicies)),
                                                dtype=np.float32)
        
        otzu_union = save_file.create_dataset(name='otzu_union',
                                            shape=(len(indicies)),
                                            dtype=np.float32)
        
        mse_values = save_file.create_dataset(name='mse_values',
                                                shape=(len(indicies),),
                                                dtype=np.float32)
        
        mae_values = save_file.create_dataset(name='mae_values',
                                                shape=(len(indicies),),
                                                dtype=np.float32)
        
        
        for i, data in enumerate(tqdm(test_data_loader)):
            video_frames, norm_gt_maps, idxs = data

            
            video_frames: torch.Tensor = video_frames.to(device)
            norm_gt_maps: torch.Tensor = norm_gt_maps.to(device)
            
            with torch.no_grad():
                # run inference
                norm_pred_maps: torch.Tensor = model(video_frames)
                
                # un_normalize the pressure maps
                pred_maps = norm_pred_maps*avg_pressure
                gt_maps = norm_gt_maps*avg_pressure

                # calculate the loss
                mae_losses = torch.abs(pred_maps - gt_maps).mean(dim=(1,2)).cpu().numpy()
                mse_losses = ((pred_maps - gt_maps)**2).mean(dim=(1,2)).cpu().numpy()

                # calculate the center of pressure
                x_locations = torch.arange(256).to(device)
                y_locations = torch.arange(256).to(device)

                pred_x_center = torch.sum(torch.sum(pred_maps, dim=2)*x_locations, dim=1)/torch.sum(pred_maps, dim=(1,2))
                pred_y_center = torch.sum(torch.sum(pred_maps, dim=1)*y_locations, dim=1)/torch.sum(pred_maps, dim=(1,2))

                gt_x_center = torch.sum(torch.sum(gt_maps, dim=2)*x_locations, dim=1)/torch.sum(gt_maps, dim=(1,2))
                gt_y_center = torch.sum(torch.sum(gt_maps, dim=1)*y_locations, dim=1)/torch.sum(gt_maps, dim=(1,2))

                # calculate the total force
                pred_total_force = torch.sum(pred_maps, dim=(1,2)).cpu().numpy()
                gt_total_force = torch.sum(gt_maps, dim=(1,2)).cpu().numpy()

                # calculate the intersection and union of the otzu thresholded images
                # first, convert to uint16 for openCV to work
                pred = ((65535/pred_maps.max())*pred_maps).cpu().numpy().astype(np.uint16)

                # run the otzu thresholding, using multiprocessing
                otzu_single = lambda x: cv2.threshold(x, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                with Pool() as p:
                    otzu_maps = p.map(otzu_single, [pred[i] for i in range(pred.shape[0])])

                otzu_maps = torch.tensor(np.stack(otzu_maps), dtype=torch.bool).to(device)

                # calculate the intersection and union
                intersection = torch.logical_and(otzu_maps, gt_maps.bool())
                union = torch.logical_or(otzu_maps, gt_maps.bool())
            
            # save the results
            indicies[i*batch_size:(i+1)*batch_size] = idxs.cpu().numpy()
            gt_pressure_map[i*batch_size:(i+1)*batch_size] = gt_maps.cpu().numpy()
            pred_pressure_map[i*batch_size:(i+1)*batch_size] = pred_maps.cpu().numpy()

            gt_center_of_pressure[i*batch_size:(i+1)*batch_size] = torch.stack((gt_x_center, gt_y_center), dim=1).cpu().numpy()
            pred_center_of_pressure[i*batch_size:(i+1)*batch_size] = torch.stack((pred_x_center, pred_y_center), dim=1).cpu().numpy()

            mse_values[i*batch_size:(i+1)*batch_size] = mse_losses
            mae_values[i*batch_size:(i+1)*batch_size] = mae_losses

            gt_total_force[i*batch_size:(i+1)*batch_size] = gt_total_force
            pred_total_force[i*batch_size:(i+1)*batch_size] = pred_total_force

            otzu_intersection[i*batch_size:(i+1)*batch_size] = intersection.cpu().numpy().sum(dim=(1,2))
            otzu_union[i*batch_size:(i+1)*batch_size] = union.cpu().numpy().sum(dim=(1,2))



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


def OtzuIOU(pred:np.ndarray, target:np.ndarray) -> Tuple[float, float]:
    # first, convert to uint16 for openCV to work
    pred = ((65535/pred.max())*pred).astype(np.uint16)

    pred_mask = cv2.threshold(pred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    intersection = np.logical_and(pred_mask, target)
    union = np.logical_or(pred_mask, target)
    
    return intersection, union


def center_of_pressure(pressure_map:np.ndarray) -> Tuple[float, float]:
    """Calculate the center of pressure of a pressure map"""
    x_locations = np.arange(pressure_map.shape[0])
    y_locations = np.arange(pressure_map.shape[1])
    
    x_center = np.sum(np.sum(pressure_map, axis=1)*x_locations)/np.sum(pressure_map)
    y_center = np.sum(np.sum(pressure_map, axis=0)*y_locations)/np.sum(pressure_map)
    
    return x_center, y_center


def center_of_pressure_distance(pred:np.ndarray, target:np.ndarray, sensor_size_mm = 41) -> float:
    pred_center = center_of_pressure(pred)
    target_center = center_of_pressure(target)

    distance = np.sqrt((pred_center[0] - target_center[0])**2 + (pred_center[1] - target_center[1])**2)

    return distance*sensor_size_mm/pred.shape[0]

def total_force_error(pred:np.ndarray, target:np.ndarray) -> float:
    return np.abs(np.sum(pred) - np.sum(target))
    


        
        

if __name__ == '__main__':
    checkpoint_path = 'data/12_hr_100_0/trained_models/cluster_run_0/checkpoints/checkpoint_71.pt'
    data_path = 'data/12_hr_100_0/hours_11_to_12.hdf5'
    save_folder = 'data/12_hr_100_0/trained_models/cluster_run_0/eval_71/'
    save_name = "11_to_12.hdf5"
    save_path = os.path.join(save_folder, save_name)
    
    run_inference(checkpoint_path, data_path, save_path)
    # print_mse_mae(save_path)

    # plot_inference_results(save_path, os.path.join(save_folder, "plots"), num_samples=5)

                    