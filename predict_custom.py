import torch
from PIL import Image
from model import CNN
import pickle

# Use GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# Load saved transform
with open('transform.pkl', 'rb') as f:
    transform = pickle.load(f)

# Load model and move to device
model = CNN().to(device)
model.load_state_dict(torch.load('cifar_model.pth', map_location=device))
model.eval()

# Load and preprocess custom image
img_path = 'custom_images/bird1.jpg'  # replace with your image path
image = Image.open(img_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)  # move image to device

# Predict
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print(f"Predicted class: {classes[predicted.item()]}")
