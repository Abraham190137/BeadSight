import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
from typing import Tuple
from torchvision import transforms

from typing import List, Tuple, Dict

from matplotlib import pyplot as plt

import time
import shutil
from tqdm import tqdm


def in_contact(forces: np.ndarray, cutoff_low:float=10, cutoff_high:float=50) -> List[bool]:
    """
    Determine which timesteps are in contact:
    For a point to be in contact, it must have a force greater than 5g.
    However, this leads to a few false positives, so we will also check to
    make sure any point that is in contact is "connected" to another point
    that has over 20g of force (definatly in contact)
    """
    contact: List[int] = [] # -1 if not in contact, 1 if in contact, 0 if unknown (above 5, below 20)
    contact.append(-1) # the first point is never in contact
    for i in range(1, len(forces)):

        # if the point is inbetween the cutoffs, its is an unknown
        if forces[i] > cutoff_low and forces[i] < cutoff_high:
            if contact[i-1] == 1: # if the previous point was in contact, this point is also in contact
                contact.append(1)
            else:
                contact.append(0) # if the previous point was not in contact, this point is unknown

        else: # if the force is outside the cutoffs, we can determine if it is in contact
            if forces[i] >= cutoff_high:
                new_val = 1
            elif forces[i] <= cutoff_low:
                new_val = -1
            else:
                raise Exception("Something VERY wrong happened")
            
            contact.append(new_val)

            # Now propagate the new value to the previous unknown points:
            for j in range(1, i):
                if contact[j-i] == 0:
                    contact[j-i] = new_val
                else:
                    break

    contact_bool = [val == 1 for val in contact]
    return contact_bool


def get_valid_indices(hdf5_file:str, window_size:int=15) -> List[int]:
    valid_indices = []
    with h5py.File(hdf5_file, 'r') as data:
        valid = data['valid'][:]
        force = data['forces'][:]
        position = data['position'][:]
        sensor_size = data.attrs['sensor_size'][:]

    # use the force to determine which points are in contact:
    contact = in_contact(force)
    
    for i in range(len(valid)):
        if i < window_size - 1:
            continue # skip the first few frames that don't have enough history

        # check to make sure the press isn't off of the sensor (shouldn't be necessary)
        if position[i][0] < -sensor_size[0]/2 \
            or position[i][0] > sensor_size[0]/2 \
            or position[i][1] < -sensor_size[1]/2 \
            or position[i][1] > sensor_size[1]/2:
            print(f"Dataset Error - Position off sensor at index {i}")
            continue

        # if the point is otherwise valid and in contact, add it to the list
        if valid[i] and contact[i]:
            valid_indices.append(i)
    return valid_indices

def get_press_indicies(hdf5_file:str, window_size:int=15) -> List[List[int]]:
    valid_indicies = get_valid_indices(hdf5_file, window_size)

    with h5py.File(hdf5_file, 'r') as data:
        positions = data['position'][:]

    # loop through the valid indicies, separating them into different presses
    press_position = positions[valid_indicies[0]]
    press_indicies = []
    current_press = []
    for idx in valid_indicies:
        if np.allclose(positions[idx], press_position):
            current_press.append(idx)
        else:
            press_indicies.append(current_press)
            current_press = [idx]
            press_position = positions[idx]

    return press_indicies


class PressureMapMaker:
    def __init__(self, 
                 image_size:Tuple[int, int], 
                 surface_size:Tuple[float,float], 
                 contact_radius:float,
                 force_unit:str = "g",
                 out_unit:str = "Pa",
                 dist_unit:str = "mm"):
        
        self.image_size = image_size
        self.surface_size = surface_size

        assert abs(image_size[0]/image_size[1] - surface_size[0]/surface_size[1]) < 0.01, "Image and surface sizes must have the same aspect ratio"

    
        # convert the force to the correct units
        if force_unit == "g":
            self.force_mult =  9.81 / 1000
        elif force_unit == "N":
            self.force_mult = 1.0
        else:
            raise ValueError("force_unit must be 'g' or 'N'")
        
        # get the contact radius in pixels:
        contact_radius_pix = round(contact_radius * image_size[0] / surface_size[0])

    
    
        # convert the radius to the correct units
        if dist_unit == "mm":
            self.contact_radius = contact_radius/1000
        elif dist_unit == "m":
            self.contact_radius = contact_radius
        else:
            raise ValueError("dist_unit must be 'mm' or 'm'")
    
        
        if out_unit == "Pa" or out_unit == "N/m^2":
            self.out_mult = 1.0
        elif out_unit == "kPa":
            self.out_mult = 0.001
        else:
            raise ValueError("out_unit must be 'Pa' or 'kPa'")
        
        # create a circle template, which can be used to quickly create the pressure map
        self.circle_template = np.zeros([2*(image_size[0] + contact_radius_pix) + 1, 2*(image_size[1] + contact_radius_pix) + 1], dtype=bool)
        center = (image_size[0] + contact_radius_pix, image_size[1] + contact_radius_pix)

        for i in range(self.circle_template.shape[0]):
            for j in range(self.circle_template.shape[1]):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                if dist < contact_radius_pix:
                    self.circle_template[i,j] = True
                else:
                    self.circle_template[i,j] = False

        self.template_center = center

    def get_bool_map(self, contact_pos:Tuple[float,float]) -> np.ndarray[bool]:
        # get the x and y positions, as a fraction of the surface size
        x_pos = 0.5 + (contact_pos[0] / self.surface_size[0])
        y_pos = 0.5 + (contact_pos[1] / self.surface_size[1])

        # get the circle template:
        center_i = round((1 - y_pos) * self.image_size[0])
        center_j = round((1 - x_pos) * self.image_size[1])

        adjust_i = self.template_center[0] - center_i
        adjust_j = self.template_center[1] - center_j

        # adjust the template to the correct position
        adj_template = self.circle_template[adjust_i:adjust_i+self.image_size[0], adjust_j:adjust_j+self.image_size[1]]

        return adj_template
    
    def pressure_from_force(self, force:float) -> float:
        return force*self.force_mult / (np.pi * self.contact_radius**2) * self.out_mult
        
        

class BeadSightDataset(Dataset):
    def __init__(self, 
                 hdf5_file:str, 
                 indicies:List[int],
                 pixel_mean:Tuple[float,float,float],
                 pixel_std:Tuple[float,float,float],
                 average_force:float,
                 train: bool,
                 rot_and_flip:bool=None,
                 window_size:int=15,
                 image_noise_std:float=0.0,
                 process_images:bool=True
                 ):
        
        print('begin dataset init')
        self.hdf5_file = hdf5_file
        self.window_size = window_size
        self.indices = indicies
        self.train = train
        self.image_noise_std = image_noise_std
        self.process_images = process_images
        
        if rot_and_flip is None:
            self.rot_and_flip = train
        else:
            self.rot_and_flip = rot_and_flip

        # get the image size:
        with h5py.File(hdf5_file, 'r') as data:
            image_size = data['images'].shape[1:3]
            sensor_size = data.attrs['sensor_size']
            contact_radius = data.attrs['contact_radius']
            force_unit = data.attrs['force_unit']
            dist_unit = data.attrs['dist_unit']


        print('begin pressure map maker')
        self.pressure_mapper = PressureMapMaker(image_size, 
                                                sensor_size, 
                                                contact_radius, 
                                                force_unit=force_unit, 
                                                dist_unit=dist_unit)

        self.image_normalize = transforms.Normalize(mean=pixel_mean, std=pixel_std)

        avg_map = self.pressure_mapper.get_bool_map((0,0)).astype(np.float32)*self.pressure_mapper.pressure_from_force(average_force)
        self.avg_pressure = np.mean(avg_map)

        self.meta_data = {"average_force": average_force,
                          "pixel_mean": pixel_mean,
                          "pixel_std": pixel_std,
                          "image_size": image_size,
                          "sensor_size": sensor_size,
                          "contact_radius": contact_radius,
                          "force_unit": force_unit,
                          "dist_unit": dist_unit}


    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, input_idx):
        with h5py.File(self.hdf5_file, 'r') as data:
            idx = self.indices[input_idx]

            # load the data
            images = data['images'][idx-self.window_size+1 : idx+1]
            force = data['forces'][idx]
            contact_pos = data['position'][idx]

        # get the pressure map
        pressure_mask = self.pressure_mapper.get_bool_map(contact_pos)
        norm_pressure = self.pressure_mapper.pressure_from_force(force)/self.avg_pressure

        # do the flip and rotation in numpy so that we get views instead of copies
        if self.rot_and_flip: # apply random rotation and flip - default is to apply only during training
            rot = np.random.randint(0, 3) # random rotation
            images = np.rot90(images, rot, (1,2))
            pressure_mask = np.rot90(pressure_mask, rot, (0,1))

            if torch.rand(1) > 0.5: # random flip
                images = np.flip(images, (1,))
                pressure_mask = np.flip(pressure_mask, (0,))
        
        pressure_map = torch.from_numpy(pressure_mask.astype(np.float32))*norm_pressure

        images = torch.from_numpy(images.copy()) # Need to copy to avoid negative strides

        # change the order of the dimensions: t, h, w, c -> t, c, h, w
        images = images.permute(0,3,1,2)

        if self.process_images:
            images = self.image_processing(images)

        return images, pressure_map, idx
    
    def image_processing(self, images:torch.Tensor) -> torch.Tensor:
        """
        Process the image tensor to be ready for the model. Can be run durring 
        dataloading of after the images are sent to the GPU. 
        Before the images are sent to the GPU decreases GPU compute
        After the images are sent to the GPU decreases .to(device) time
        """
        images = self.image_normalize(images.float())
        if self.image_noise_std > 0:
            images = torch.normal(images, self.image_noise_std)

        return images
    
def replay_data(hdf5_file:str):
    import cv2
    with h5py.File(hdf5_file, 'r') as data:
        pixel_mean = data.attrs['pixel_mean']
        pixel_std = data.attrs['pixel_std']
        average_force = data.attrs['average_force']
        max_force = max(data['forces'])
        force_data = data['forces'][:]

    plt.figure()
    plt.plot(np.arange(len(force_data)), force_data)
    plt.show()

    print(f"Pixel Mean: {pixel_mean}")
    print(f"Pixel Std: {pixel_std}")
    print(f"Average Force: {average_force}")
    print(f"Max Force: {max_force}")
    
    max_norm = max_force / average_force

    print('begin checking valid indices')

    indices = get_valid_indices(hdf5_file)

    dataset = BeadSightDataset(hdf5_file=hdf5_file,
                               indicies=indices,
                               pixel_mean=pixel_mean,
                               pixel_std=pixel_std,
                               average_force=average_force,
                               train=False,)
    
    for i in range(len(dataset)):
        images, pressure_map, idx = dataset.__getitem__(i)
        image = images[-1].permute(1,2,0).numpy()
        pressure_map = pressure_map.numpy()

        un_norm_image = image*pixel_std + pixel_mean

        print(i)
        cv2.imshow('Image', un_norm_image/255)
        cv2.imshow('Pressure Map', pressure_map/max_norm)
        cv2.waitKey(1)

def decompress_h5py(in_file:str, out_file:str):
    # makes a copy of the file, decompressing the images

    # first, copy the file
    shutil.copy(in_file, out_file)

    with h5py.File(out_file, 'r+') as out_data:
        images_shape = out_data['images'].shape
        del out_data['images']
        uncompressed_images = out_data.create_dataset(name = 'images', 
                                                      shape = images_shape,
                                                      chunks = (1, images_shape[1], images_shape[2], images_shape[3]), 
                                                      dtype = np.uint8)
        with h5py.File(in_file, 'r') as in_data:
            images = in_data['images']
            for i in tqdm(range(images.shape[0]), desc="Decompressing Images"):
                uncompressed_images[i] = images[i]
            
        
if __name__ == "__main__":
    replay_data("/home/aigeorge/research/BeadSight/data/initial_test_34/processed_data.hdf5")