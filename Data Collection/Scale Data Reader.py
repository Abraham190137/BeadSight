import serial
import time
import numpy as np
import os

recording = []
save_path = "output/weight_readings_0.csv"
i = 0
while os.path.exists(save_path):
    save_path = save_path[:-len(str(i)) + 4] + str(i+1) + ".csv"
    i += 1
with open(save_path, 'w') as save_file:
    with serial.Serial('COM6', 57600, timeout=20) as ser:
        start = False
        last_line = None
        while True:
            line = ser.readline()   # read a '\n' terminated line
            if line == last_line:
                continue
            last_line = line
            if not start:
                if line.decode()[:5] == "start":
                    start_time = time.time()
                    start = True
                continue
            try:
                measurement = float(line.decode()[:-1])%10000
                if measurement > 9000:
                    measurement -= 10000
            except:
                continue
            elapsed_time = time.time() - start_time
            recording.append((measurement, elapsed_time))
            save_file.write(str(np.round(measurement, 2)) + "," + str(np.round(elapsed_time, 4)) + "\n")

