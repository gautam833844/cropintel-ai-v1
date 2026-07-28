"""
Constants & Validation Rules for CropIntel AI
=============================================
"""

INPUT_VALIDATION = {
    'nitrogen': {'min': 0, 'max': 150, 'type': 'float'},
    'phosphorus': {'min': 0, 'max': 150, 'type': 'float'},
    'potassium': {'min': 0, 'max': 210, 'type': 'float'},
    'temperature': {'min': -50, 'max': 60, 'type': 'float'},
    'humidity': {'min': 0, 'max': 100, 'type': 'float'},
    'ph': {'min': 0, 'max': 14, 'type': 'float'},
    'rainfall': {'min': 0, 'max': 500, 'type': 'float'}
}

REQUIRED_FIELDS = list(INPUT_VALIDATION.keys())
FEATURE_NAMES = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
