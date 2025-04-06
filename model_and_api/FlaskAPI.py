import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_cors import CORS



# Load class label
TRAIN_DIR = 'C:\\Users\\sugam\\Desktop\\mri-images\\model_and_api\\Training'  # replace with your actual path
class_labels = sorted(os.listdir(TRAIN_DIR))

# load model
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)


# debuging
print("Loading model from:", MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("Model loaded successfully.")

print("Loading class labels from:", TRAIN_DIR)
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Train directory not found at {TRAIN_DIR}")

class_labels = sorted(os.listdir(TRAIN_DIR))
print("Class labels:", class_labels)



# Create Flask app
app = Flask(__name__)

CORS(app)  # Allow CORS for all routes and origins


def preprocess_image(image, image_size=128):
    img = load_img(image, target_size=(image_size, image_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions)

    predicted_label = class_labels[predicted_class_index]

    if predicted_label.lower() == 'notumor':
        result = "No Tumor"
    else:
        result = f"Tumor Found: {predicted_label}"

    return {
        "result": result,
        "confidence": f"{confidence_score * 100:.2f}%"
    }

# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    print("Request method:", request.method)
    print("Content-Type:", request.content_type)
    print("Request.files:", request.files)
    print("Request.form:", request.form)
    print("Request.data:", request.data)
    print("Request.headers:", request.headers)

    # Check if 'image' is part of the request
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save image to uploads folder
    image_path = os.path.join('uploads', image_file.filename)
    image_file.save(image_path)

    try:
        output = predict_image(image_path)  # your image classification function
        os.remove(image_path)
        return jsonify(output)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

        
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, use_reloader=False)