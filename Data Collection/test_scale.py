import serial
import time
import numpy as np

start_time = time.time()
with serial.Serial('/dev/ttyACM0', 57600, timeout=2) as ser:
    start = False
    last_line = None
    while True:
        line = ser.readline()   # read a '\n' terminated line
        # print(line)
        try:
            measurement = float(line.decode()[:-1])
        except:
            print("Error reading line:", line)
            continue
        current_time = time.strftime('%H:%M:%S.') + str(int(time.time() % 1 * 1000)).zfill(3)
        seconds = time.time() - start_time
        print(measurement, seconds, current_time)