import cv2
import matplotlib.pyplot as plt

INFO_FILE = "test_info_file.csv"
VIDEO_FILE = "out5_09_09_58.mp4"
GCODE_FILE = "testing Gcode 4-5.g"

PIXEL_SIZE = 13.67/455
FINGER_SIZE_MM = 14
FINGER_SIZE_PIX = int(FINGER_SIZE_MM/PIXEL_SIZE)

def select_frame(cap):
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    start_time = input("Enter start time: ")
    start_frame = int(float(start_time)*fps)
    print(start_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fc = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)
        plt.show()
        command = input("frame " + str(fc) +". To stop, type 'stop'. \n"
                        + "To move to a different frame, enter frame number."
                        + "To move to next fame, press enter.")
        if command == "stop":
            break
        if command != "":
            try:
                fc = int(command)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fc)
            except:
                print("Invalid input. Proceeding to next frame.")
        else:
            fc += 1
    return fc

# cap = cv2.VideoCapture(VIDEO_FILE)

# print('Select calibration frame, the frame where the leds dim')
# calibration_frame = select_frame(cap)
# print('Select start frame, the frame to start the first push.')
# start_frame = select_frame(cap)
# cap.release()
# end_frame = select_frame(cap)
# cap.release()

calibration_frame = 199
start_frame = 855
end_frame = 23100

points = []
with open(GCODE_FILE, 'r') as gcode_file:
    gcode = gcode_file.read().split('\n')

for line in gcode:
    # The gcode command lines are of the form "G1 Z## X## Y##"
    # Note: The first press is NOT recorded as a test, but is included.
    # It's used to find the center of the sensor of calibration
    if not "G1" in line:
        continue
    x_idx = line.rfind("X")
    y_idx = line.rfind("Y")
    f_idx = line.rfind("F")

    x = float(line[x_idx+1:y_idx-1])
    y = float(line[y_idx+1:f_idx-1])

    point = (x, y)
    if len(points) == 0 or points[-1] != point:
        points.append(point)    

with open(INFO_FILE, 'w') as info_file:
    info_file.write(str(calibration_frame) + "," + str(start_frame) + "," + str(end_frame) + '\n')
    info_file.write(str(points[0][0]) + "," + str(points[0][1]) + '\n')
    points.pop(0)
    info_file.write(str(len(points))+ '\n') # number of TEST points, does not include the first point (center)
    for point in points:
        info_file.write(str(point[0]) + "," + str(point[1]) + '\n')



