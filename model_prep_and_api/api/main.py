import io
import logging
import os
from typing import Dict
import warnings

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import onnxruntime as ort
import uvicorn

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "model.onnx"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")

class_labels = ["glioma", "meningioma", "notumor", "pituitary"]

# Load ONNX Inference Session
if os.path.exists(MODEL_PATH):
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Loaded ONNX model session from {MODEL_PATH}")
else:
    session = None
    input_name = None
    output_name = None
    print(f"Warning: Model file not found at {MODEL_PATH}.")

app = FastAPI(
    title="Brain Tumor Prediction API",
    description="FastAPI service for Brain Tumor Detection and Classification using ONNX Runtime",
    version="1.0.0",
)

allowed_origins = ["*"] if CORS_ORIGINS == "*" else [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_valid_mri(image: Image.Image) -> tuple[bool, str]:
    """
    Validates whether the input image is a genuine Brain MRI scan.
    Rejects colored photos (cats, dogs, landscapes, selfies, etc.)
    """
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32)

    # 1. Color Variance / Saturation Check (MRI scans are grayscale images where R == G == B)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    rg_diff = np.abs(r - g)
    gb_diff = np.abs(g - b)
    br_diff = np.abs(b - r)
    color_std = float(np.mean(rg_diff + gb_diff + br_diff))

    if color_std > 12.0:
        return False, "The uploaded image is not a valid Brain MRI scan (color photo detected)."

    # 2. Dark Background Ratio Check (MRI scans have dark borders around skull tissue)
    brightness = np.mean(arr, axis=2)
    dark_pixels_ratio = float(np.mean(brightness < 50))

    if dark_pixels_ratio < 0.10:
        return False, "The uploaded image lacks a standard brain MRI scan background."

    return True, "Valid MRI scan"


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def predict_image(image_bytes: bytes) -> Dict[str, float | str]:
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model is not loaded."
        )

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format or corrupted file."
        )

    # Validate whether image is a genuine Brain MRI scan
    valid_mri, reason = is_valid_mri(image)
    if not valid_mri:
        return {
            "result": "Invalid Image",
            "confidence": 0.0,
            "error": reason
        }

    # Preprocessing: Resize to 128x128, normalize [0, 1], transpose to (1, 3, 128, 128)
    resized_img = image.resize((128, 128))
    img_np = np.array(resized_img).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    tensor_input = np.expand_dims(img_np, axis=0)

    outputs = session.run([output_name], {input_name: tensor_input})[0]
    probabilities = softmax(outputs)[0]
    predicted_idx = int(np.argmax(probabilities))
    confidence_score = float(probabilities[predicted_idx])

    predicted_label = class_labels[predicted_idx]
    result = "No Tumor" if predicted_label.lower() == "notumor" else f"Tumor Found: {predicted_label}"
    confidence_value = float(confidence_score * 100)

    return {
        "result": result,
        "confidence": round(confidence_value, 2),
    }


@app.get("/health", summary="Health Check")
def health_check():
    return {"status": "ok"}


@app.post("/predict", summary="Predict Brain Tumor from MRI Image")
async def predict(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded."
        )

    contents = await image.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    output = predict_image(contents)
    return JSONResponse(content=output)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
