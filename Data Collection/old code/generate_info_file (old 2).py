import cv2
import matplotlib.pyplot as plt
import json


DIR = "C:/Research/Tactile Sensor/Tactile_Sensor/Data Collection/data/7-12_test_1/"
VIDEO_FILE = DIR + "2023-07-12 02-11-46.mp4"
GCODE_FILE = DIR + "data_collection_2_7-12.g"
CENTER_POINT = [160, 85]

### IMPORTANT ### 
# The code assumes that in the video, as the x location increases the finger moves to the right, and as the 
# y location increases the finger moves down. If this is not the case, the code will need to be modified.

# SENSOR_SIZE = 
# FINGER_SIZE_MM = 12.5
# FINGER_SIZE_PIX = int(FINGER_SIZE_MM/PIXEL_SIZE)

def select_frame(cap):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    start_time = input("Enter start time: ")
    cv2.namedWindow("frame")
    start_frame = int(float(start_time)*fps)
    print(start_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fc = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("frame", frame)
        cv2.waitKey(1) 
        command = input("frame " + str(fc) +". To stop, type 'stop'. \n"
                        + "To move to a different frame, enter frame number."
                        + "To move to next fame, press enter. ")
        if command == "stop":
            break
        if command != "":
            try:
                fc = int(command)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fc)
            except:
                print("Invalid input. Proceeding to next frame. ")
        else:
            fc += 1

    time_stamp = input('Plese enter the timestamp: ')
    cv2.destroyAllWindows()
    return fc, time_stamp

cap = cv2.VideoCapture(VIDEO_FILE)

print('Select start frame, the frame to start the first push.')
start_frame, time_stamp = select_frame(cap)
cap.release()


points = []
with open(GCODE_FILE, 'r') as gcode_file:
    gcode = gcode_file.read().split('\n')

for line in gcode:
    # The gcode command lines are of the form "G1 Z## X## Y##"
    # Note: The first press is NOT recorded as a test, but is included.
    # It's used to find the center of the sensor of calibration
    if (not "G1" in line) or ("calibration" in line) or ("move up" in line):
        continue
    x_idx = line.rfind("X")
    y_idx = line.rfind("Y")
    z_idx = line.rfind("Z")

    x = float(line[x_idx+1:y_idx-1])
    y = float(line[y_idx+1:z_idx-1])

    point = (x, y)
    if len(points) == 0 or points[-1] != point:
        points.append(point)    

save_dict = {}
save_dict['start_frame'] = start_frame
save_dict['start_timestamp'] = time_stamp
save_dict['points'] = points
save_dict['video_name'] = VIDEO_FILE.split('/')[-1]
save_dict['center_point'] = CENTER_POINT

with open(DIR + 'info.json', 'w') as json_file:
    json.dump(save_dict, json_file)



