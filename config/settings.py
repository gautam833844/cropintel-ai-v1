"""
Settings and Path Configurations for CropIntel AI
=================================================
Centralized path definitions and application settings using pathlib.Path.
"""

from pathlib import Path
import os

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Core Directories
CONFIG_DIR = BASE_DIR / 'config'
SRC_DIR = BASE_DIR / 'src'
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
GENERATED_DATA_DIR = DATA_DIR / 'generated'
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
TEMPLATES_DIR = BASE_DIR / 'templates'
STATIC_DIR = BASE_DIR / 'static'
SCRIPTS_DIR = BASE_DIR / 'scripts'
DEPLOYMENT_DIR = BASE_DIR / 'deployment'
TESTS_DIR = BASE_DIR / 'tests'
NOTEBOOKS_DIR = BASE_DIR / 'notebooks'

# File Paths
ORIGINAL_DATASET_PATH = RAW_DATA_DIR / 'Crop_recommendation.csv'
LARGE_DATASET_PATH = GENERATED_DATA_DIR / 'Crop_recommendation_large.csv'
MODEL_PATH = ARTIFACTS_DIR / 'crop_model.pkl'
SCALER_PATH = ARTIFACTS_DIR / 'scaler.pkl'

# Model & Prediction Settings
CONFIDENCE_TEMPERATURE = 0.6
CONFIDENCE_BOOST_FACTOR = 1.15
MIN_CONFIDENCE = 75.0
