from torchvision import transforms
import torch
import cv2
from copy import deepcopy
from Unet_model import UNet
import numpy as np
import PIL.Image as Image
from tqdm import tqdm as tmdq


video_file = "/data/robot videos/force_15_apple.mp4"
model_path = 'best_model.pth'
save_path = "data/robot videos/force_15_apple_out.npy"
start_frame = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet().to(device)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to 256x256
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
])

cap = cv2.VideoCapture(video_file)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

entire_video = torch.zeros((num_frames-start_frame, 3, 256, 256), dtype=torch.float32)

# load video frames
for i in tmdq(range(num_frames-start_frame)):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(Image.fromarray(frame))
    entire_video[i] = frame

pressure_maps = np.zeros((num_frames-start_frame-15, 256, 256))

for i in tmdq(range(num_frames-start_frame - 15)):
    video_frames = torch.concat(entire_video[i:i+15], dim=0).clone().unsqueeze(0)
    with torch.no_grad():
        predictions = model(video_frames.to(device))*1000
        pressure_maps[i] = predictions.squeeze().cpu().numpy()


np.save(save_path, pressure_maps)
cap.release()

