"""
This file contains functions for running inference on a dataset and printing/plotting the results.
"""

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

def otzu_single(image:np.ndarray) -> np.ndarray:
    """
    Run otzu thresholding on a single image. This function is used for multiprocessing.
    """
    # convert to uint16 for openCV to work
    image = ((65535/image.max())*image).astype(np.uint16)
    return cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def run_inference(checkpoint_path: str,
                  data_path: str,
                  save_path: str,
                  indicies: str,
                  batch_size: int = 64,
                  otzu_batch_size: int = 1024):
    """
    Run inference on the dataset and save the results to a file.
    """
    # Load the dataset
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    window_size:int = checkpoint['window_size']
    pixel_mean: float = checkpoint['pixel_mean']
    pixel_std: float = checkpoint['pixel_std']
    average_force: float = checkpoint['data_loader_info']['average_force']
    
    model = UNet(window_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if indicies == "test": # If no indicies are provided, use the saved test indicies
        indicies = checkpoint['test_indices']

    elif indicies == "all": # If all indicies are provided, use all the indicies
        indicies = get_valid_indices(data_path, window_size)

    else:
        raise ValueError("indicies must be 'test' or 'all'")

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
    
    # get the max force for each press:
    press_indicies = get_press_indicies(data_path, window_size)
    with h5py.File(data_path, 'r') as data_file:
        forces = data_file['forces'][:]
        sensor_dim = data_file.attrs['sensor_size'] # mm
        sensor_area = sensor_dim[0]*sensor_dim[1]/(1000**2) # m^2
        press_max_pressures = []
        for idx_list in press_indicies:
            press_max_force = np.max(forces[idx_list])
            press_max_pressures.append(dataset.pressure_mapper.pressure_from_force(press_max_force))
    
    with h5py.File(save_path, 'w') as save_file:
        save_file.attrs['checkpoint_path'] = checkpoint_path
        save_file.attrs['data_path'] = data_path
        save_file.attrs['avg_pressure'] = avg_pressure
        save_file.attrs['sensor_size'] = sensor_dim
        
        #copy the checkpoint info the the attributes, excluding the model and optimizer:

        # save_file.attrs['checkpoint_info'] = checkpoint_info

        save_file.create_dataset(name='press_max_pressures',
                                 data=press_max_pressures)

        idx_data = save_file.create_dataset(name='indicies',
                                                shape=(len(indicies),),
                                                dtype=np.int32)


        gt_pressure_map = save_file.create_dataset(name='gt_pressure_map', 
                                                shape=(len(indicies), 256, 256),
                                                chunks=(1, 256, 256), 
                                                dtype=np.float32,
                                                compression=9)
        
        gt_pressure_values = save_file.create_dataset(name='gt_pressure_values',
                                                shape = (len(indicies),),
                                                dtype=np.float32)
        
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
                mae_values[i*batch_size:(i+1)*batch_size] = torch.abs(pred_maps - gt_maps).mean(dim=(1,2)).cpu().numpy()
                mse_values[i*batch_size:(i+1)*batch_size] = ((pred_maps - gt_maps)**2).mean(dim=(1,2)).cpu().numpy()

                # calculate the center of pressure
                x_locations = torch.arange(256).to(device)
                y_locations = torch.arange(256).to(device)

                pred_x_center = torch.sum(torch.sum(pred_maps, dim=2)*x_locations, dim=1)/torch.sum(pred_maps, dim=(1,2))
                pred_y_center = torch.sum(torch.sum(pred_maps, dim=1)*y_locations, dim=1)/torch.sum(pred_maps, dim=(1,2))

                gt_x_center = torch.sum(torch.sum(gt_maps, dim=2)*x_locations, dim=1)/torch.sum(gt_maps, dim=(1,2))
                gt_y_center = torch.sum(torch.sum(gt_maps, dim=1)*y_locations, dim=1)/torch.sum(gt_maps, dim=(1,2))

                # calculate the total force
                pred_total_force[i*batch_size:(i+1)*batch_size] = torch.mean(pred_maps, dim=(1,2)).cpu().numpy()*sensor_area # total force, N
                gt_total_force[i*batch_size:(i+1)*batch_size] = torch.mean(gt_maps, dim=(1,2)).cpu().numpy()*sensor_area # total force, N
            
            # save the results
            idx_data[i*batch_size:(i+1)*batch_size] = idxs.cpu().numpy()
            gt_pressure_map[i*batch_size:(i+1)*batch_size] = gt_maps.cpu().numpy()
            gt_pressure_values[i*batch_size:(i+1)*batch_size] = gt_maps.amax(dim=(1,2)).cpu().numpy()
            pred_pressure_map[i*batch_size:(i+1)*batch_size] = pred_maps.cpu().numpy()

            gt_center_of_pressure[i*batch_size:(i+1)*batch_size] = torch.stack((gt_x_center, gt_y_center), dim=1).cpu().numpy()
            pred_center_of_pressure[i*batch_size:(i+1)*batch_size] = torch.stack((pred_x_center, pred_y_center), dim=1).cpu().numpy()


        # finally, run otzu thresholding, which uses a different batch size (much larger)

        intersections = []
        unions = []

        for batch_idx in tqdm(range(0, n_samples, otzu_batch_size)):
            start_idx = batch_idx
            end_idx = min(batch_idx + batch_size, n_samples)

            pred = pred_pressure_map[start_idx:end_idx]
            gt = gt_pressure_map[start_idx:end_idx]

            # run the otzu thresholding, using multiprocessing
            with Pool() as p:
                otzu_maps = p.map(otzu_single, [pred[i] for i in range(pred.shape[0])])

            otzu_maps = np.stack(otzu_maps)
            pred_bin = otzu_maps.astype(bool)
            gt_bin = gt.astype(bool)

            intersection = np.logical_and(pred_bin, gt_bin)
            union = np.logical_or(pred_bin, gt_bin)

            intersections.append(intersection.sum(axis=(1,2)))
            unions.append(union.sum(axis=(1,2)))

        intersections = np.concatenate(intersections)
        unions = np.concatenate(unions)

        save_file.create_dataset(name='otzu_intersection',
                                    data=intersections,
                                    dtype=np.float32)
        
        save_file.create_dataset(name='otzu_union',
                                    data=unions,
                                    dtype=np.float32)
    


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
                 

def print_eval_metrics(results_path: str):
    with h5py.File(results_path, 'r') as results_file:
        mse_values = results_file['mse_values'][:]
        mae_values = results_file['mae_values'][:]
        otzu_intersections = results_file['otzu_intersection'][:]
        otzu_unions = results_file['otzu_union'][:]
        gt_total_force = results_file['gt_total_force'][:]
        pred_total_force = results_file['pred_total_force'][:]
        gt_center_of_pressure = results_file['gt_center_of_pressure'][:]
        pred_center_of_pressure = results_file['pred_center_of_pressure'][:]
        press_max_pressures = results_file['press_max_pressures'][:]
        avg_max_pressure = np.mean(press_max_pressures)
        avg_pressure = results_file.attrs['avg_pressure']


        print(f'Average Pressure: {avg_pressure} Pa')
        print(f"Max Max Pressure: {np.max(press_max_pressures)} Pa")
        print(f"Average Max Pressure: {avg_max_pressure} Pa")

        print(f'MAE: {mae_values.mean()} Pa')
        print(f'Percent MAE: {100*mae_values.mean()/avg_max_pressure}%')
        print(f'MSE: {mse_values.mean()} Pa^2')
        print(f'RMSE: {np.sqrt(mse_values.mean())} Pa')
        print(f'Percent RMSE: {100*np.sqrt(mse_values.mean())/avg_max_pressure}%')

        print(f"Average Force Error: {100*np.mean(np.abs(gt_total_force - pred_total_force))/np.mean(gt_total_force)}%")
        
        # only calculate IOU for pressure maps with a total force greater than 2N
        mask = gt_total_force[:-1] > 2
        print('mask intersection over union:', np.mean(otzu_intersections[mask]/otzu_unions[mask]))
        
        distances = np.sqrt(np.sum((gt_center_of_pressure - pred_center_of_pressure)**2, axis=1))
        avg_dist_error = distances.mean()
        sensor_size = 41 # mm
        image_size = results_file['gt_pressure_map'].shape[1]
        print(f"Average distance error: {avg_dist_error} pix, {avg_dist_error*sensor_size/image_size} mm")
        
def error_location_analysis(inference_path: str):
    grid_size = 4
    # make 3x3 grid for analysis:
    total_errors = np.zeros((grid_size, grid_size))
    num_presses = np.zeros((grid_size, grid_size))

    # Load the generated pressure maps
    with h5py.File(inference_path, 'r') as data_file:
        all_gen_pressure_maps = data_file['pred_pressure_map'][:]/1000 #kPa
        all_gt_pressure_maps = data_file['gt_pressure_map'][:]/1000 #kPa

    print(f'all_gen_pressure_maps shape: {all_gen_pressure_maps.shape}')
    print(f'all_gt_pressure_maps shape: {all_gt_pressure_maps.shape}')

    for i in tqdm(range(all_gen_pressure_maps.shape[0])):
        if all_gt_pressure_maps[i].sum() == 0:
            continue

        for x in range(grid_size):
            for y in range(grid_size):
                if all_gt_pressure_maps[i][int(x*256/grid_size):int((x+1)*256/grid_size), int(y*256/grid_size):int((y+1)*256/grid_size)].sum() > 0:
                    num_presses[x, y] += 1
                    total_errors[x, y] += np.mean(np.abs(all_gen_pressure_maps[i] - all_gt_pressure_maps[i]))
                

    matplotlib.rcParams['font.size'] = 16  # Adjust to change default font size for all text
    plt.figure(figsize=(5, 5))
    plt.imshow(total_errors/num_presses, vmin=0)
    cbar = plt.colorbar(shrink=0.8)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Error (kPa)', rotation=270, labelpad=20)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('Error_vs_Location.jpg', dpi = 300)
    print('max error:', np.max(total_errors/num_presses))
       

def make_horizon_graph(inference_paths: List[str]):
    mae_values = []
    gt_total_forces = []
    IOUs = []
    avg_distances = []
    n_evals = len(inference_paths)
    for inference_path in inference_paths:
        with h5py.File(inference_path, 'r') as f:
            mae_values.append(f['mae_values'][:].mean())
            gt_total_forces.append(f['gt_total_force'][:])
            # only calculate IOU for pressure maps with a max pressure greater than 2N
            mask = gt_total_forces[-1][:-1] > 2
            IOUs.append(np.mean(f['otzu_intersection'][mask]/f['otzu_union'][mask]))
            distances = np.sqrt(np.sum((f['gt_center_of_pressure'][:] - f['pred_center_of_pressure'][:])**2, axis=1))
            avg_distances.append(distances.mean())

    # plot the normalized MAE and IOU
    normalized_MAE = mae_values/mae_values[0]*100
    normalized_IOU = IOUs/IOUs[0]*100
    normalized_dist = avg_distances/avg_distances[0]*100

    # print the average and final percentage values:
    print("Average Percentage Values:")
    print(f"MAE: {np.mean(normalized_MAE)}")
    print(f"IOU: {np.mean(normalized_IOU)}")
    print(f"Distance: {np.mean(normalized_dist)}")

    print("Final Percentage Values:")
    print(f"MAE: {normalized_MAE[-1]}")
    print(f"IOU: {normalized_IOU[-1]}")
    print(f"Distance: {normalized_dist[-1]}")
    plt.figure()
    # plt.title("Normalized Performace Over Time")
    plt.plot(range(0, n_evals), normalized_MAE, label='MAE Error')
    plt.plot(range(0, n_evals), normalized_IOU, label='IOU Errror')
    plt.plot(range(0, n_evals), normalized_dist, label='Distance Error')
    plt.plot(range(0, n_evals), np.ones(n_evals)*100, '--', c = 'gray')
    plt.xlabel('Hours')
    plt.ylabel('Percent of Initial Value')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    checkpoint_path = "data/12_hr_75/trained_models/hours_1_to_3_same_as_paper_0/checkpoints/checkpoint_99.pt"
    data_path = 'data/12_hr_75/hours_3_to_4.hdf5'
    save_folder = 'data/12_hr_75/trained_models/hours_1_to_3_same_as_paper_0/eval'
    save_name = "3_to_4.hdf5"
    save_path = os.path.join(save_folder, save_name)
    
    run_inference(checkpoint_path, data_path, save_path, indicies='all')

    print_eval_metrics(save_path)



                    