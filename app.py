from flask import Flask, render_template, request, redirect, url_for
import os
import keras
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('bird_sound_recognition_model.h5')

# Recreate the label encoder
bird_species = ['acafly', 'acowoo', 'aldfly', 'ameavo', 'amecro', 'amegfi']
label_encoder = LabelEncoder()
label_encoder.fit(bird_species)

# Function to preprocess the uploaded audio file
def preprocess_audio_file(file_path, sr=32000, duration=5):
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)  # Add channel dimension for CNN input
    mel_spec_db = np.repeat(mel_spec_db, 3, axis=-1)    # Convert to 3 channels for ResNet50
    return mel_spec_db

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)
        
        mel_spec = preprocess_audio_file(file_path)
        mel_spec = np.expand_dims(mel_spec, axis=0)  # Add batch dimension
        
        prediction = model.predict(mel_spec)
        predicted_class = np.argmax(prediction, axis=1)
        
        bird_species = label_encoder.inverse_transform(predicted_class)[0]
        print(file.filename)
        
        return render_template('result.html', bird_species=bird_species, filename=file.filename)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True)
