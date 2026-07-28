#!/usr/bin/env python3
"""Test the API fix for field names"""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.api.index import app
client = app.test_client()

# Test data with correct field names (matching form)
test_data = {
    'nitrogen': 50,
    'phosphorus': 50,
    'potassium': 50,
    'temperature': 25,
    'humidity': 70,
    'ph': 6.5,
    'rainfall': 100
}

print("=" * 60)
print("Testing API /api/predict endpoint")
print("=" * 60)
print(f"\nSending test data:")
print(json.dumps(test_data, indent=2))

response = client.post(
    '/api/predict',
    data=json.dumps(test_data),
    content_type='application/json'
)

print(f"\nResponse Status: {response.status_code}")
result = json.loads(response.data)
print(f"Response Body:")
print(json.dumps(result, indent=2))

if response.status_code == 200:
    print("\n" + "=" * 60)
    print("✓ SUCCESS - API is working!")
    print("=" * 60)
    print(f"Predicted Crop: {result.get('crop')}")
    print(f"Confidence: {result.get('confidence')}%")
else:
    print("\n" + "=" * 60)
    print("✗ ERROR - API returned error")
    print("=" * 60)
    print(f"Error: {result.get('error')}")
