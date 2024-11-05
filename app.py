from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO
import io
from PIL import Image
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = YOLO('model/best.pt')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        image = cv2.imread(image_path)
        results = model(image)
        
        annotated_image = results[0].plot()
        
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return render_template('result.html', img_data=img_str)

    return 'Invalid file format. Only JPG, PNG, and JPEG are allowed.'

if __name__ == '__main__':
    app.run(debug=True)
