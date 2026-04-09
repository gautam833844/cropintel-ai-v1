#!/bin/bash

# Vercel Build Script - Train ML Model
# This script runs during Vercel deployment to generate required model files

echo "=========================================="
echo "Building Hydroponic Crop Recommendation System"
echo "=========================================="

# Check if models exist
if [ ! -f "crop_model.pkl" ] || [ ! -f "scaler.pkl" ]; then
    echo "Models not found. Training ML model..."
    echo "This may take 2-5 minutes..."
    
    # Run training script
    python train_model.py
    
    if [ $? -eq 0 ]; then
        echo "✓ Model training completed successfully"
    else
        echo "✗ Model training failed"
        exit 1
    fi
else
    echo "✓ Models already exist, skipping training"
fi

echo "✓ Build completed"
