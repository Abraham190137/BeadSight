import serial
import time
import numpy as np
import os

recording = []
save_path = "C:/Research/Tactile Sensor/Tactile_Sensor/Data Collection/output/weight_readings_7-12_0.csv"
i = 0
while os.path.exists(save_path):
    save_path = save_path[:-(len(str(i)) + 4)] + str(i+1) + ".csv"
    print('new save path:', save_path)
    i += 1

start_time = time.time()
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
                    start = True
                continue
            try:
                measurement = float(line.decode()[:-1])%10000
                if measurement > 9000:
                    measurement -= 10000
            except:
                continue
            current_time = time.strftime('%H:%M:%S.') + str(int(time.time() % 1 * 1000)).zfill(3)
            seconds = time.time() - start_time
            recording.append((measurement, current_time))
            print(measurement, seconds, current_time)
            save_file.write(str(np.round(measurement, 2)) + "," + str(np.round(seconds, 4)) + ',' + current_time +  "\n")

