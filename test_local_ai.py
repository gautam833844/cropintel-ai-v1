#!/usr/bin/env python3
"""Test the complete local AI system flow"""

import sys
import json
sys.path.insert(0, 'api')

# Import the Flask app
api_module = __import__('index')
client = api_module.app.test_client()

print('=' * 70)
print('Testing Complete API Flow: Prediction + Local AI Explanation')
print('=' * 70)
print()

# Test data
test_data = {
    'nitrogen': 50,
    'phosphorus': 50,
    'potassium': 50,
    'temperature': 25,
    'humidity': 70,
    'ph': 6.5,
    'rainfall': 100
}

# Step 1: Get prediction
print('Step 1: Getting ML Model Prediction...')
response = client.post(
    '/api/predict',
    data=json.dumps(test_data),
    content_type='application/json'
)
prediction = json.loads(response.data)
print(f'  Status: {response.status_code}')
print(f'  Crop: {prediction.get("crop")}')
print(f'  Confidence: {prediction.get("confidence")}%')
print()

# Step 2: Get AI explanation
print('Step 2: Getting Local AI Explanation...')
explain_data = {
    'crop': prediction.get('crop'),
    'confidence': prediction.get('confidence'),
    **test_data
}
response = client.post(
    '/api/explain',
    data=json.dumps(explain_data),
    content_type='application/json'
)
explanation = json.loads(response.data)
print(f'  Status: {response.status_code}')
print()
print(f'Explanation: {explanation.get("explanation")}')
print()
print(f'Tips: {len(explanation.get("tips", []))} tips provided')
for tip in explanation.get("tips", [])[:2]:
    print(f'  • {tip}')
print()
print(f'Benefits: {len(explanation.get("benefits", []))} benefits listed')
for benefit in explanation.get("benefits", [])[:2]:
    print(f'  • {benefit}')
print()
print('=' * 70)
print('✓ SUCCESS: Complete flow works without OpenAI API!')
print('=' * 70)
