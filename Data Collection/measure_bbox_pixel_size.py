import cv2

VIDEO_FILE = "out5_09_09_58.mp4"
cap = cv2.VideoCapture(VIDEO_FILE)
ret, frame = cap.read()

bbox = cv2.selectROI(frame)
print(bbox[2], bbox[3])