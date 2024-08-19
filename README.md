# BeadSight
This repo contains the code from the paper, "BeadSight: An Inexpensive Tactile Sensor Using Hydro-Gel Beads"

## Data Collection:
This folder contains all of the code used for data collection and processing. It contains:
 - Arduino code (in the Arduino folder) which was used to collect force data. Scale Data Reader.py is used to control the Arduino over serial link and to save the recorded forces and timestamps.
 - Gcode generation (generate_g_code.py), which was used to create the GCode used for data collection.
 - Data Processing (raw_data_process.py), which used force data and the GCode data to generate pressure-map data (which is used in training) and to split the recorded video into individual presses.
 - Pressure Map Extraction (unpack_force_data.py), which converted the saved force data (.pkl file) to a pressure map. The pressure map is much more memory-intensive, which is why the pickle files are saved instead.
 - Example Usage (create_demo_force_video.py), in the form of a script to unpack the .pkl files to create and display a pressure map over time video.

## BeadSight Unet: 
This folder contains the code for the Unet Model used in the Bead Sight paper
 - U-Net model (Unet_model.py): Pytorch implementation of the UNet model
 - Data Processing (data_processing.py): Post-Processing of the data before use in the UNet model training 
 - Training and Testing (model_training.py): Training and Testing of the BeadSight UNet model, using the recorded data.
 - Inference (inference.py): Code for implementing the trained UNet model (loaded from saved weights) to process images collected during robot operation.

Note to reviewers: I just re-submitted the paper (wanted to prioritize getting that out) and am currently working on cleaning the code base. I will push the updated code this week (by 8/21)
