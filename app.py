from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import CNN

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

device = torch.device('cpu')
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

model = CNN()
model.load_state_dict(torch.load('cifar_model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    filename = None
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            image = Image.open(save_path).convert('RGB')
            image = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(image)
                _, pred = torch.max(output, 1)
                prediction = classes[pred.item()]
    return render_template('index.html', filename=filename, prediction=prediction)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        image = Image.open(save_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)
            prediction = classes[pred.item()]
        return jsonify({'prediction': prediction})
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
