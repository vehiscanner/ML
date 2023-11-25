from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2  # Import OpenCV for video processing
import os

# Load model
model = tf.keras.models.load_model('model1.h5')

app = Flask(__name__)

# Define folder to save recorded videos
UPLOAD_FOLDER = 'hasilrecord'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

counter = 1  # Inisialisasi counter untuk penamaan file
class_names = ['car', 'bycyle', 'motorcycle']  # Ganti dengan kelas yang sesuai

@app.route('/predict', methods=['POST'])
def process_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video provided'}), 400
        
        video = request.files['video']
        
        # Menyimpan video ke dalam folder hasilrecord dengan penamaan video_01, video_02, dan seterusnya
        global counter
        video_name = f'video_{str(counter).zfill(2)}.mp4'
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
        video.save(video_path)
        
        # Process the video and save individual frames
        video_capture = cv2.VideoCapture(video_path)
        frame_counter = 1
        
        # Initialize a dictionary to store prediction counts for each class
        class_counts = {class_name: 0 for class_name in class_names}
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Preprocess the frame for prediction (resize, normalize, etc.)
            frame = cv2.resize(frame, (224, 224))
            img_array = np.array(frame)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            # Perform prediction on the frame
            predictions = model.predict(img_array)
            predictionsmax = np.argmax(predictions)
            predicted_class = class_names[predictionsmax]
            
            # Update the count for the predicted class
            class_counts[predicted_class] += 1
            
            # You can handle the predictions for each frame here
            
            frame_counter += 1
        
        # Calculate and print the count of each class
        for class_name, count in class_counts.items():
            print(f'{class_name}: {count}')
        
        return jsonify({'message': 'Video processed successfully', 'counts': class_counts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)