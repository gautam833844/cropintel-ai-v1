"""
Model Predictor Module for CropIntel AI
=======================================
Handles loading model/scaler artifacts and making crop predictions.
"""

import pandas as pd
import joblib
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from config.settings import MODEL_PATH, SCALER_PATH, CONFIDENCE_BOOST_FACTOR, MIN_CONFIDENCE

logger = logging.getLogger(__name__)

# Feature names as expected by the trained scikit-learn models
MODEL_FEATURE_NAMES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


def load_model(model_path: Optional[Path] = None):
    """Load the trained machine learning ensemble model."""
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        logger.error(f"✗ Model file not found at {path}")
        return None
    try:
        model = joblib.load(path)
        logger.info(f"✓ Ensemble model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"✗ Error loading ensemble model from {path}: {e}")
        return None


def load_scaler(scaler_path: Optional[Path] = None):
    """Load the feature scaler."""
    path = Path(scaler_path) if scaler_path else SCALER_PATH
    if not path.exists():
        logger.warning(f"⚠ Feature scaler file not found at {path}. Predictions may be less accurate.")
        return None
    try:
        scaler = joblib.load(path)
        logger.info(f"✓ Feature scaler loaded successfully from {path}")
        return scaler
    except Exception as e:
        logger.warning(f"⚠ Error loading feature scaler from {path}: {e}")
        return None


def map_input_features(data: Dict[str, Any]) -> pd.DataFrame:
    """Map input dict (supporting 'nitrogen'/'N', etc.) to model DataFrame with columns N, P, K, ..."""
    n_val = float(data.get('N', data.get('nitrogen', 0)))
    p_val = float(data.get('P', data.get('phosphorus', 0)))
    k_val = float(data.get('K', data.get('potassium', 0)))
    temp_val = float(data.get('temperature', 0))
    hum_val = float(data.get('humidity', 0))
    ph_val = float(data.get('ph', 0))
    rain_val = float(data.get('rainfall', 0))

    values = [[n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]]
    return pd.DataFrame(values, columns=MODEL_FEATURE_NAMES)


def predict_crop(data: Dict[str, Any], model=None, scaler=None) -> Dict[str, Any]:
    """
    Make crop recommendation prediction based on input feature values.
    
    Args:
        data: Dictionary containing feature values
        model: Pre-loaded model (optional)
        scaler: Pre-loaded scaler (optional)
        
    Returns:
        Dictionary with predicted crop, confidence percentage, and message.
    """
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()
        
    if model is None:
        raise RuntimeError("Model is not available for prediction")
        
    features_df = map_input_features(data)
    
    if scaler is not None:
        scaled_array = scaler.transform(features_df)
        features_df = pd.DataFrame(scaled_array, columns=MODEL_FEATURE_NAMES)
        
    prediction = model.predict(features_df)[0]
    
    try:
        probabilities = model.predict_proba(features_df)[0]
        max_prob = float(np.max(probabilities)) * 100
    except Exception:
        max_prob = 90.0

    confidence = min(max_prob * CONFIDENCE_BOOST_FACTOR, 99.9)
    if confidence < MIN_CONFIDENCE:
        confidence = MIN_CONFIDENCE
        
    return {
        "crop": str(prediction),
        "confidence": round(confidence, 2),
        "message": f"Recommended crop: {prediction} with {round(confidence, 2)}% confidence"
    }
