from torchvision import transforms
import torch
import cv2
from Unet_model import UNet
from typing import List
from defisheye import Defisheye

# model params
CHECKPOINT_PATH = 'best_model.pth'
SAVE_PATH = "data/robot videos/force_15_apple_out.npy"

# video settings
WIDTH = 1280
HEIGHT = 1024
FPS = 30

DEWARPING_PARAMS = {
    "fov": 180,
    "pfov": 120,
    "xcenter": None,
    "ycenter": None,
    "radius": None,
    "angle": 0,
    "dtype": "linear",
    "format": "fullframe"
}
CROP = (142, 892, 120, 870)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
window_size:int = checkpoint['window_size']
pixel_mean: List[float] = checkpoint['pixel_mean']
pixel_std: List[float] = checkpoint['pixel_std']
avg_pressure: float = checkpoint['avg_pressure']
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to 256x256
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize(mean=pixel_mean, std=pixel_std) # Normalize the images
])

# Open the beadsight camera. Alternatively, you can open a video file for testing.
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # helps increase the fps

ret, frame = cap.read()
if not ret:
    raise Exception("Failed to open the camera")

defisheye = Defisheye(frame, **DEWARPING_PARAMS)
defisheye.calculate_conversions()

frame_buffer = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # preprocess the frame
    frame = torch.tensor(frame).to(device)
    frame = defisheye.unwarp(frame)
    frame = frame[CROP[0]:CROP[1], CROP[2]:CROP[3]]

    # add the frame to the buffer:
    frame_buffer.append(frame)
    if len(frame_buffer) < window_size:
        continue # keep filling the buffer until it reaches the window size
    
    # once the buffer is full, we can start processing the frames
    video_frames = torch.stack(frame_buffer).clone().permute(0, 3, 1, 2) # t, h, w, c -> t, c, h, w
    video_frames = transform(video_frames)

    with torch.no_grad():
        # predict the pressure maps, and multiply by the average pressure to get the actual pressure values
        predictions = model(video_frames)*avg_pressure # Pa
    
    # show the pressure map:
    pressure_map = predictions.squeeze().cpu().numpy()
    cv2.imshow('Pressure Map', pressure_map)
    cv2.waitKey(1)

    # finally, remove the oldest frame from the buffer
    frame_buffer.pop(0)

