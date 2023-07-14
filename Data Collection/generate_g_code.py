import numpy as np

G_CODE_FILE = "Tactile_Sensor\Data Collection\data_collection_3.g"

g_code:str = """; 7/12 Testing code

M140 S0 T0; set temp of bed
M104 S0 T0; set temp of extruder 1
M104 S0 T1; set temp of extruder 2
G90; use absolute positioning\n"""

X_RANGE = [150, 180]
Y_RANGE = [70, 100]
Z_RANGE = [109, 115]
Z_UP = 120

NUM_PRESSES = 500

g_code += f"\nG1 X0 Y0 Z{Z_UP} F700; move up\n"

# Calibaration presses:
for _ in range(3):
    for x_pos, y_pos in [(X_RANGE[0]-10, Y_RANGE[0]-10), (X_RANGE[0]-10, Y_RANGE[1]+10), (X_RANGE[1]+10, Y_RANGE[0]-10), (X_RANGE[1]+10, Y_RANGE[1]+10)]:
        g_code_line  = f"\nG1 X{x_pos} Y{y_pos} Z{Z_UP} F700; calibration motion"
        g_code_line += f"\nG1 X{x_pos} Y{y_pos} Z{Z_RANGE[0]} F200; calibration motion"
        g_code_line += "\nG04 P1000"
        g_code_line += f"\nG1 X{x_pos} Y{y_pos} Z{Z_UP} F700; calibration motion\n"
        g_code += g_code_line

for x_pose, y_pose in [(X_RANGE[0], Y_RANGE[0]), (X_RANGE[0], Y_RANGE[1]), (X_RANGE[1], Y_RANGE[0]), (X_RANGE[1], Y_RANGE[1])]:
    g_code_line  = f"\nG1 X{x_pose} Y{y_pose} Z{Z_UP} F700; calibration motion"
    g_code_line += f"\nG1 X{x_pose} Y{y_pose} Z{Z_RANGE[0]} F200; calibration motion"
    g_code_line += "\nG04 P1000"
    g_code_line += f"\nG1 X{x_pose} Y{y_pose} Z{Z_UP} F700; calibration motion\n"
    g_code += g_code_line

for i in range(NUM_PRESSES):
    x_pos = np.random.uniform(X_RANGE[0], X_RANGE[1])
    y_pos = np.random.uniform(Y_RANGE[0], Y_RANGE[1])
    z_pos = np.random.uniform(Z_RANGE[0], Z_RANGE[1])

    g_code_line  = f"\nG1 X{x_pos} Y{y_pos} Z{Z_UP} F700; move to position {i}"
    g_code_line += f"\nG1 X{x_pos} Y{y_pos} Z{z_pos} F200; move to position {i}"
    g_code_line += "\nG04 P1000"
    g_code_line += f"\nG1 X{x_pos} Y{y_pos} Z{Z_UP} F700; move to position {i}\n"
    g_code += g_code_line


with open(G_CODE_FILE, 'w') as file:
    file.write(g_code)