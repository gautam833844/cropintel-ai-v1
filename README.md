# CropIntel AI - Smart Crop Recommendation System

CropIntel AI is an enterprise-grade, high-performance crop recommendation platform powered by a VotingClassifier ensemble ML model (RandomForest, GradientBoosting, and ExtraTrees).

## Project Architecture

```text
cropintel-ai-v1/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .env.example
├── app.py                  # Main Flask Web Application
├── train.py                # Master Training Orchestrator
├── predict.py              # Prediction Utility CLI
├── config/                 # App Settings & Constants
│   ├── __init__.py
│   ├── settings.py
│   └── constants.py
├── src/                    # Python Source Code
│   ├── api/                # Flask / Serverless API Functions
│   ├── preprocessing/      # Data Scalers & Preprocessors
│   ├── models/             # Model Training & Inference Logic
│   ├── ai/                 # Local AI Knowledge Base
│   └── utils/              # Helper Utilities
├── data/                   # Data Directory
│   ├── raw/                # Original Datasets
│   ├── processed/          # Processed Data
│   └── generated/          # Generated / Augmented Datasets
├── artifacts/              # Model Artifacts (.pkl)
├── templates/              # HTML Templates
├── static/                 # Static Assets (CSS/JS)
├── tests/                  # Test Suite
├── docs/                   # Documentation
├── scripts/                # Utility & Build Scripts
├── deployment/             # Deployment Configurations (Dockerfile, vercel.json)
└── notebooks/              # Jupyter Notebooks
```

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Training the ML Model
```bash
python train.py
```

### 3. Running the Web Application
```bash
python app.py
```
Open `http://localhost:5000` in your web browser.

### 4. Running Predictions via CLI
```bash
python predict.py
```

### 5. Running Tests
```bash
python tests/test_api_endpoint.py
python tests/test_local_ai.py
python tests/test_model_loading.py
```
