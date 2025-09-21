from flask import Flask, render_template, request
import torch
from PIL import Image
import pickle
from model import CNN
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model and transform
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

model = CNN().to(device)
model.load_state_dict(torch.load('cifar_model.pth', map_location=device))
model.eval()

with open('transform.pkl', 'rb') as f:
    transform = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_path = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                prediction = classes[predicted.item()]

    return render_template('index.html', prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
