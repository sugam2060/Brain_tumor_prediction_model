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
import torch
from torchvision import transforms
import uvicorn

from model_def import BrainTumorVGG16

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "model.pth"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")

class_labels = ["glioma", "meningioma", "notumor", "pituitary"]
device = torch.device("cpu")

# Image transformation pipeline (Resize to 128x128, convert to tensor [0, 1])
image_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Initialize PyTorch Model
model = BrainTumorVGG16(num_classes=len(class_labels), freeze_features=True)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded PyTorch model weights from {MODEL_PATH}")
else:
    print(f"Warning: Model file not found at {MODEL_PATH}.")

model.to(device)
model.eval()

app = FastAPI(
    title="Brain Tumor Prediction API",
    description="FastAPI service for Brain Tumor Detection and Classification using PyTorch (VGG16)",
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


def predict_image(image_bytes: bytes) -> Dict[str, float | str]:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format or corrupted file."
        )

    tensor_img = image_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence_score, predicted_idx = torch.max(probabilities, dim=0)

    predicted_label = class_labels[predicted_idx.item()]
    result = "No Tumor" if predicted_label.lower() == "notumor" else f"Tumor Found: {predicted_label}"
    confidence_value = float(confidence_score.item() * 100)

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
