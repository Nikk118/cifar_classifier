import pickle
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Save the transform
with open('transform.pkl', 'wb') as f:
    pickle.dump(transform, f)

print("Transform saved as transform.pkl")
