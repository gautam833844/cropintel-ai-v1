"""
Utility Helpers for CropIntel AI
================================
"""

import logging
from pathlib import Path
from config.settings import BASE_DIR, ARTIFACTS_DIR, MODEL_PATH, SCALER_PATH


def setup_logger(name: str = __name__) -> logging.Logger:
    """Set up and return a configured logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name)


def check_artifacts_exist() -> bool:
    """Check if model and scaler artifacts exist."""
    return Path(MODEL_PATH).exists() and Path(SCALER_PATH).exists()
