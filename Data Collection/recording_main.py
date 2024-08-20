import cv2
import time
import os
import serial
from typing import List, Tuple
import numpy as np
import multiprocessing
import json


def video_logger(log_folder:str,
                 stop_event: multiprocessing.Event,
                 chunk_size:int=30*60*60, # save a new "chunk" every hour
                 height:int=1024,
                 width:int=1280,
                 fps:int=30,
                 debug_print:bool = False):

    # create the log folder
    frame_number = 0
    video_path = f"{log_folder}/video_{frame_number//chunk_size}.avi"
    log_path = f"{log_folder}/video_log_{frame_number//chunk_size}.txt"

    assert not os.path.exists(video_path), f"File already exists: {video_path}"
    assert not os.path.exists(log_path), f"File already exists: {log_path}"

    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Using XVID codec for AVI format
    video_file = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    log_file = open(log_path, 'w')

    # Open the webcam
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    previous_time = time.time()
    start_time = previous_time

    while not stop_event.is_set():
        # Read a frame from the webcam
        ret, frame = cap.read()
        cur_time = time.time()
        if not ret:
            break

        log_file.write(f"{cur_time},{frame_number},{video_path}\n")
        log_file.flush()
        video_file.write(frame)
        frame_number += 1
        
        if debug_print:
            print('\nframe_number:', frame_number)
            print('dt:', cur_time - previous_time)
            print('Avergae FPS:', frame_number / (cur_time - start_time))
            previous_time = cur_time
            print('frame shape:', frame.shape)

        if frame_number % chunk_size == 0:
            # save the video and log files
            video_file.release()
            log_file.close()
            video_path = f"{log_folder}/video_{frame_number//chunk_size}.avi"
            log_path = f"{log_folder}/video_log_{frame_number//chunk_size}.txt"
            assert not os.path.exists(video_path), f"File already exists: {video_path}"
            assert not os.path.exists(log_path), f"File already exists: {log_path}"
            video_file = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            log_file = open(log_path, 'w')

    # Release the webcam, video file, and log file
    cap.release()
    video_file.release()
    log_file.close()
    print('Video logger stopped successfully')


def force_logger(log_folder:str, 
                 stop_event: multiprocessing.Event,
                 serial_port:str = '/dev/ttyACM0'):

    assert os.path.exists(log_folder), f"Folder does not exist: {log_folder}"
    log_file = open(f"{log_folder}/force_log.txt", 'w') 
    last_read_time = time.time()   

    with serial.Serial(serial_port, 57600, timeout=2) as ser:
        start = False
        while not stop_event.is_set():
            if time.time()-last_read_time > 10:
                raise ValueError("No data read in 10 seconds")
            line = ser.readline()   # read a '\n' terminated line
            try:
                measurement = float(line.decode()[:-1])
            except:
                print("Error reading line:", line)
                continue
            cur_time = time.time()
            last_read_time = cur_time
            log_file.write(f"{cur_time},{measurement}\n")
            log_file.flush()
            
    log_file.close()
    print('Force logger stopped successfully')


class PrinterControl:
    def __init__(self, log_folder:str, serial_port:str) -> None:
        self.log_folder = log_folder
        self.log_file = open(f"{log_folder}/printer_log.txt", 'w')
        self.ser = serial.Serial(serial_port, 115200) 
        self.closed = False

    def send_command(self, command:str, rec_x:float=-1, rec_y:float=-1, delay:float=1):
        self.ser.write((command + "\n").encode())
        print(command)
        cur_time = time.time()
        self.log_file.write(f"{cur_time},{rec_x},{rec_y},{command}\r") # need to use \r, because the command can contain \n
        self.log_file.flush()
        time.sleep(delay)

    def close(self):
        self.ser.close()
        self.log_file.close()
        self.closed = True

    def __del__(self):
        if not self.closed:
            self.close()

# message for one press action
def press_message(x_loc: float, y_loc: float, z_down:float, z_up:float):
    return f"G0 X{x_loc} Y{y_loc} Z{z_up} F700; move to position\n" + \
           f"G0 X{x_loc} Y{y_loc} Z{z_down} F200; press\n" + \
           "G04 P1000; pause\n" + \
           f"G0 X{x_loc} Y{y_loc} Z{z_up} F700; release"

def run_exeriment(log_folder:str, 
                  run_time:float,
                  x_range:Tuple[float, float],
                  y_range:Tuple[float, float],
                  z_range:Tuple[float, float],
                  z_clear:float = 160):

    # initalize the loggers:
    stop_event = multiprocessing.Event()
    video_logger_process = multiprocessing.Process(target=video_logger, args=(log_folder, stop_event), daemon=True)
    video_logger_process.start()
    force_logger_process = multiprocessing.Process(target=force_logger, args=(log_folder, stop_event), daemon=True)
    force_logger_process.start()
    printer = PrinterControl(log_folder, '/dev/ttyUSB0')

    try:
        # initalize the printer:
        printer.send_command("M140 S0 T0; set temp of bed", delay = 1)
        printer.send_command("M104 S0 T0; set temp of extruder 1", delay = 1)
        printer.send_command("M104 S0 T1; set temp of extruder 2", delay = 1)
        printer.send_command("G90; use absolute positioning", delay = 1)
        printer.send_command("G28; home all axes", delay = 45)
        printer.send_command(f"G0 X0 Y0 Z{z_clear} F700; move to clear position", delay = 30)
        printer.send_command(f"G0 X{x_range[0]-10} Y{y_range[0]-10} Z{z_clear} F700; calibration motion", delay = 30)

        calib_offset = 10
        # Calibaration presses:
        for x_pos, y_pos in [(x_range[0]-calib_offset, y_range[0]-calib_offset), (x_range[0]-calib_offset, y_range[1]+calib_offset), (x_range[1]+calib_offset, y_range[0]-calib_offset), (x_range[1]+calib_offset, y_range[1]+calib_offset)]:
            printer.send_command(press_message(x_pos, y_pos, z_range[0], z_clear), -1, -1, 15)

        for x_pos, y_pos in [(x_range[0], y_range[0]), (x_range[0], y_range[1]), (x_range[1], y_range[0]), (x_range[1], y_range[1])]:
            printer.send_command(press_message(x_pos, y_pos, z_range[0], z_clear), -1, -1, 15)

        # begin the real testing:
        start_time = time.time()
        press_num = 0
        while time.time() - start_time < run_time:
            x_pos = np.random.uniform(x_range[0], x_range[1])
            y_pos = np.random.uniform(y_range[0], y_range[1])
            z_pos = np.random.uniform(z_range[0], z_range[1])

            printer.send_command(press_message(x_pos, y_pos, z_pos, z_clear), x_pos, y_pos, 10)

        print("Experiment completed successfully")

    except:
        print("An error occured, stopping the experiment")

    stop_event.set()
    video_logger_process.join()
    force_logger_process.join()
    printer.close()

if __name__ == "__main__":
    save_dir = "Data Collection/data"
    save_name = "5 hr data collection - 100 my bag"
    run_time = 5*60*60 # number of seconds to run for.
    center = (151.8, 108.9)
    radius = 13.75
    x_range = (center[0] - radius, center[0] + radius)
    y_range = (center[1] - radius, center[1] + radius)
    z_range = (127, 128)
    z_clear = 137.5

    n = 0
    while os.path.exists(f"{save_dir}/{save_name}_{n}"):
        n += 1
    folder_name = f"{save_dir}/{save_name}_{n}"
    os.mkdir(folder_name)

    # save parametes in json file:
    meta_data = {
        "resolution": (256, 256),
        "fps": 30,
        "contact_radius": 6.25,
        "sensor_size": (41, 41),
        "force_unit": "g",
        "dist_unit": "mm",
        "run_time": run_time,
        "center": center,
        "radius": radius,
        "x_range": x_range, 
        "y_range": y_range, 
        "z_range": z_range, 
        "z_clear": z_clear
    }

    with open(f"{folder_name}/experiment_parameters.json", 'w') as f:
        json.dump(meta_data, f)

    run_exeriment(log_folder=folder_name,
                  run_time=run_time,
                  x_range=x_range,
                  y_range=y_range,
                  z_range=z_range,
                  z_clear=z_clear)
    



