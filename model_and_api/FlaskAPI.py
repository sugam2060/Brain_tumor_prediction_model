import logging
import os
from typing import Dict
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").propagate = False

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "model.keras"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")

class_labels = ["glioma", "meningioma", "notumor", "pituitary"]


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = load_model(MODEL_PATH, compile=False)

app = Flask(__name__)
allowed_origins = "*" if CORS_ORIGINS == "*" else [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]
CORS(app, resources={r"/*": {"origins": allowed_origins}})

os.makedirs(UPLOAD_DIR, exist_ok=True)


def preprocess_image(image_path: str, image_size: int = 128):
    img = load_img(image_path, target_size=(image_size, image_size))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_image(image_path: str) -> Dict[str, float | str]:
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = int(np.argmax(predictions, axis=1)[0])
    confidence_score = float(np.max(predictions))

    predicted_label = class_labels[predicted_class_index]
    result = "No Tumor" if predicted_label.lower() == "notumor" else f"Tumor Found: {predicted_label}"
    confidence_value = float(confidence_score * 100)

    return {
        "result": result,
        "confidence": round(confidence_value, 2),
    }


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = os.path.basename(image_file.filename)
    image_path = os.path.join(UPLOAD_DIR, filename)
    image_file.save(image_path)

    try:
        output = predict_image(image_path)
        return jsonify(output)
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)