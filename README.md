# BeadSight
This repo contains the code from the paper, "BeadSight: An Inexpensive Tactile Sensor Using Hydro-Gel Beads"

 - Unet_model.py: Pytorch implementation of the UNet model
 - raw_data_processing.py: Processes the video files, force data, and Gcode commands recorded by recording_main, converting the data into a hdf5 file.
 - model_training.py: Training and Testing of the BeadSight UNet model, using the recorded data.
 - model_eval.py: Functions for running inference on a dataset and printing/plotting the results.
 - inference.py: Code for implementing the trained UNet model (loaded from saved weights) to process images collected during robot operation.
 - defisheye.py: Helper script to undo the fish-eye effect of the bead-sight camera's lens.
 - data_loader.py: Dataset class, along with helper functions, for using the processed beadsight training data.


## Data Collection:
This folder contains all of the code used for data collection and processing. It contains:
 - Arduino code (in the Arduino folder) which was used to collect force data. Scale Data Reader.py is used to control the Arduino over serial link and to save the recorded forces and timestamps.
 - recording_main.py, which records video from the bead sight camera (USB camera), records the current force applied, measured by the Arduino and sent over a serial connection, and controls the 3D printer to execute presses, via commands sent over a serial connection.
