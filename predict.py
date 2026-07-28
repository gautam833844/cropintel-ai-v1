#!/usr/bin/env python3
"""
CLI & Modular Prediction Utility for CropIntel AI
=================================================
Runs crop recommendation predictions using the trained ensemble model.
"""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.models.predictor import predict_crop, load_model, load_scaler


def main():
    print("=" * 60)
    print("CropIntel AI - Prediction Utility")
    print("=" * 60)

    # Example test sample
    sample_input = {
        'nitrogen': 50.0,
        'phosphorus': 50.0,
        'potassium': 50.0,
        'temperature': 25.0,
        'humidity': 70.0,
        'ph': 6.5,
        'rainfall': 100.0
    }

    if len(sys.argv) > 1:
        try:
            sample_input = json.loads(sys.argv[1])
        except Exception as e:
            print(f"Error parsing JSON argument: {e}")
            sys.exit(1)

    print("\nInput Parameters:")
    for k, v in sample_input.items():
        print(f"  • {k.capitalize()}: {v}")

    try:
        model = load_model()
        scaler = load_scaler()
        result = predict_crop(sample_input, model=model, scaler=scaler)
        print("\nPrediction Result:")
        print(f"  • Recommended Crop: {result['crop']}")
        print(f"  • Confidence: {result['confidence']}%")
        print(f"  • Summary: {result['message']}")
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
