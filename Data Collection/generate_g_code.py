import numpy as np

G_CODE_FILE = "Tactile_Sensor\Data Collection\data_collection_2.g"

g_code:str = """; 7/10 Testing code

M140 S0 T0; set temp of bed
M104 S0 T0; set temp of extruder 1
M104 S0 T1; set temp of extruder 2
G90; use absolute positioning\n"""

X_RANGE = [130, 150]
Y_RANGE = [130, 150]
Z_RANGE = [155, 160]
Z_UP = 165

NUM_PRESSES = 100
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