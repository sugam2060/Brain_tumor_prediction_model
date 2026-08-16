# 🧠 Brain Tumor Detection & Classification System (PyTorch + FastAPI + React)

An end-to-end Deep Learning web application for automatic detection and multi-class classification of brain tumors from MRI scans. Built with **PyTorch (torchvision VGG16)**, **FastAPI (Uvicorn)** backend REST API, and a modern **React + Vite** frontend interface. Fully compatible with **Python 3.12+**.

---

## 📌 Project Structure

```
Brain_tumor_prediction_model/
├── .gitattributes                # Git LFS configuration for *.pth weights
├── .gitignore                    # Ignores dataset folders & temporary build files
├── .python-version               # Python version (3.12.0)
├── README.md                     # Documentation & setup guide
│
├── model_prep_and_api/           # Parent Container Folder
│   ├── model_prep/               # Training & Dataset Prep
│   │   ├── model_def.py          # PyTorch BrainTumorVGG16 model architecture
│   │   ├── train_pytorch_model.py# PyTorch training & evaluation script (saves model.pth to api/)
│   │   ├── requirement_model_prep.txt # Training & GPU dependencies (CUDA 12.1, PyTorch, tqdm, etc.)
│   │   ├── Training/             # Training dataset (ignored in git)
│   │   ├── Testing/              # Testing dataset (ignored in git)
│   │   └── report/               # Saved evaluation charts & text report
│   │       ├── confusion_matrics.png
│   │       ├── ROC_curve.png
│   │       ├── classification_report.txt
│   │       └── model_training_history(accuracy vs loss).png
│   │
│   └── api/                      # Backend FastAPI Service (Deployable to Render)
│       ├── main.py               # FastAPI REST API for image inference
│       ├── model_def.py          # PyTorch model architecture definition
│       ├── model.pth             # Saved PyTorch model weights (Git LFS)
│       ├── requirement.txt       # Production dependencies for API (PyTorch, FastAPI, Uvicorn)
│       ├── .env.example          # Environment variables template
│       ├── .python-version       # Set to 3.12.0
│       └── uploads/              # Temporary image uploads
│
└── Mini-project-frontend/        # React + Vite Frontend UI
    ├── package.json              # NPM dependencies & build scripts
    ├── index.html                # Main HTML entry point
    ├── src/                      # React components & UI logic
    └── .env.example              # Frontend environment variables template
```

---

## 🛠️ Local Running & Training Guide

### Prerequisites
- **Python**: Version 3.12 or higher
- **Node.js**: Version 18.x or higher
- **Git & Git LFS**: For managing `model.pth` weights

---

### Step 1: Model Training (Local with GPU CUDA Support)

1. Open terminal and navigate to `model_prep_and_api/model_prep`:
   ```bash
   cd model_prep_and_api/model_prep
   ```
2. Install training & GPU dependencies:
   ```bash
   pip install -r requirement_model_prep.txt
   ```
3. Place `Training/` and `Testing/` dataset folders inside `model_prep_and_api/model_prep/`.
4. Run the training script:
   ```bash
   python train_pytorch_model.py
   ```
   *This trains the pre-trained VGG16 PyTorch model with GPU acceleration (RTX 3050), generates evaluation reports in `model_prep/report/`, and automatically saves `model.pth` into `model_prep_and_api/api/`.*

---

### Step 2: Backend Setup (FastAPI + Uvicorn)

1. Navigate to `model_prep_and_api/api`:
   ```bash
   cd model_prep_and_api/api
   ```
2. Install production API dependencies:
   ```bash
   pip install -r requirement.txt
   ```
3. Start the FastAPI server with Uvicorn:
   ```bash
   uvicorn main:app --port 5000 --reload
   ```
   *Backend API runs at `http://localhost:5000`.*
   *Interactive API Documentation (Swagger UI): `http://localhost:5000/docs`.*

---

### Step 3: Frontend Setup (React + Vite)

1. Navigate to `Mini-project-frontend`:
   ```bash
   cd Mini-project-frontend
   ```
2. Install NPM packages & start development server:
   ```bash
   npm install
   npm run dev
   ```
3. Open `http://localhost:5173` in your browser. Upload an MRI scan to view classification results!

---

## 🌐 Cloud Deployment (Render)

### Backend Service (`model_prep_and_api/api`)
1. Create a **Web Service** on [Render](https://dashboard.render.com/).
2. Set **Root Directory**: `model_prep_and_api/api`
3. Set **Runtime**: `Python 3`
4. Set **Build Command**: `pip install -r requirement.txt`
5. Set **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Set Environment Variable: `PYTHON_VERSION` = `3.12.0`

### Frontend Static Site (`Mini-project-frontend`)
1. Create a **Static Site** on Render.
2. Set **Root Directory**: `Mini-project-frontend`
3. Set **Build Command**: `npm install && npm run build`
4. Set **Publish Directory**: `dist`
5. Set Environment Variable: `VITE_API_BASE_URL` = `https://your-backend-api.onrender.com`
