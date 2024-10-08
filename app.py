from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the pre-trained model (.h5 file)
model = load_model(r'C:\Users\DELL\Desktop\AQI\aqi_prediction_model.h5')

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/')
def index():
    return render_template(r'C:\Users\DELL\Desktop\AQI\index.html') 

# Define the route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (224, 224))
        img_resized = img_resized.astype('float32') / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)
        prediction = model.predict(img_resized)
        predicted_aqi = float(prediction[0][0]) 
        return jsonify({'predicted_aqi': predicted_aqi})

if __name__ == '__main__':
    app.run(debug=True)
