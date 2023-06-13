import torch
import os
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
from Unet_model import UNet

# Load the model
model = UNet()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Define the necessary transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to 256x256
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # Normalize the images
])

def load_images(folder, transform, test_number):
    images = []
    for i in range(5):
        image_path = os.path.join(folder, f'test_{test_number}_frame_{i}.png')
        image = Image.open(image_path)
        image = transform(image)
        images.append(image)
    return torch.cat(images, dim=0)

# Process each set of inputs
for test_number in range(80, 100):  # Test numbers from 80 to 99
    images = load_images('frame/predict_video', transform, test_number)

    # Make a prediction
    with torch.no_grad():
        prediction = model(images.unsqueeze(0))

    # Apply sigmoid function to the output
    prediction = torch.sigmoid(prediction)

    # Convert tensor to PIL image
    prediction_image = transforms.ToPILImage()(prediction.squeeze(0))

    # Save the image
    prediction_image.save(f'frame/predict_pressure/test_{test_number}_frame_4_p.png')



