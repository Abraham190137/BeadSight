import pickle
import numpy as np

def unpack_data(data_file, resolution=[0, 0]):
    # load in the force data
    with open(data_file, 'rb') as file:
        data = pickle.load(file) 

    # check if the resolution of the output was set by the user:
    if resolution != [0, 0]:
        ratio = [resolution[0]/data['height'], resolution[1]/data['width']]
        data['height'] = resolution[0]
        data['width'] = resolution[1]
    else:
        # if it wasn't set, use default ratio of 1, 1
        ratio = [1, 1]

    # create the contact footprint. This is a circle (oval if resolution is non-square) representing the points that are in
    # contact with the sensor.
    touch_footprint = np.zeros([data['height'], data['width']])
    for x in range (touch_footprint.shape[1]):
        for y in range(touch_footprint.shape[0]):
            if (data['center_pxl_x']-x/ratio[1])**2 + (data['center_pxl_y']-y/ratio[0])**2 < data['contact_radius']**2:
                touch_footprint[y, x] = 1

    # multiply the foot print with the measured force (in kN/m^2) to get the output force map for each frame.
    touch_data = np.empty([len(data['forces']), data['height'], data['width']])
    for i, force in enumerate(data['forces']):
        touch_data[i, :, :] = touch_footprint*force

    return touch_data