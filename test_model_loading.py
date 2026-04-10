#!/usr/bin/env python3
"""Test script to verify model loading in the API"""

import sys
import os
sys.path.insert(0, 'api')

print("\n=== Model Loading Verification ===\n")

# Import the Flask app module
try:
    import index
    print("✓ Successfully imported api/index.py")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Check if models loaded
print(f"\nModel Status:")
print(f"  - Model loaded: {'YES' if index.model else 'NO'}")
print(f"  - Scaler loaded: {'YES' if index.scaler else 'NO'}")

if index.model:
    print(f"  - Model type: {type(index.model).__name__}")
    print(f"  - Ready for predictions: YES")

    # Test prediction
    try:
        import numpy as np
        test_input = np.array([[50, 50, 50, 25, 70, 6.5, 100]])

        if index.scaler:
            test_scaled = index.scaler.transform(test_input)
        else:
            test_scaled = test_input

        pred = index.model.predict(test_scaled)
        print(f"  - Test prediction result: {pred[0]}")
        print("\n✓ API is READY to make predictions on Vercel!")
    except Exception as e:
        print(f"  - Error during prediction: {e}")
else:
    print("\n✗ ERROR: Models are NOT loaded. Deployment will fail.")
    print("   Train_model.py must complete successfully first.")
