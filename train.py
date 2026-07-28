"""
Master Training Pipeline Orchestrator for CropIntel AI
======================================================
This script orchestrates the complete training pipeline:
1. Generate augmented training samples
2. Train the ensemble model to achieve high accuracy
3. Verify saved artifacts in artifacts/
4. Optionally start the Flask web application
"""

import sys
import subprocess
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    ORIGINAL_DATASET_PATH,
    LARGE_DATASET_PATH,
    MODEL_PATH,
    SCALER_PATH
)


def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def run_command(command: str, description: str) -> bool:
    """Run a command and handle errors."""
    print(f"\n{'─' * 80}")
    print(f"📌 {description}")
    print(f"{'─' * 80}\n")

    try:
        result = subprocess.run(command, shell=True, capture_output=False)
        if result.returncode != 0:
            print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


def main():
    print_banner("🌾 CROP RECOMMENDATION SYSTEM - MASTER TRAINING PIPELINE")

    # Step 1: Check if original dataset exists
    print("\n[STEP 1] Checking dataset availability...")
    if not ORIGINAL_DATASET_PATH.exists():
        print(f"❌ ERROR: {ORIGINAL_DATASET_PATH} not found!")
        print("   Please ensure the dataset is in data/raw/ directory.")
        sys.exit(1)
    print(f"✓ Original dataset found at {ORIGINAL_DATASET_PATH}")

    # Step 2: Generate large dataset
    print("\n[STEP 2] Generating augmented training samples...")
    if not run_command(f'python "{BASE_DIR / "scripts" / "generate_dataset.py"}"', 'Dataset generation'):
        print("⚠ Warning: Dataset generation may have had issues. Continuing anyway...")

    # Check if large dataset was created
    if LARGE_DATASET_PATH.exists():
        print(f"✓ Large dataset generated successfully at {LARGE_DATASET_PATH}")
    else:
        print("⚠ Large dataset not found. Training script will generate it on-the-fly.")

    # Step 3: Train the ensemble model
    print("\n[STEP 3] Training ensemble model...")
    if not run_command(f'python "{BASE_DIR / "src" / "models" / "train_model.py"}"', 'Model training'):
        print("\n❌ CRITICAL: Model training failed!")
        print("   Please check the error messages above.")
        sys.exit(1)

    # Check if model was saved
    if not MODEL_PATH.exists():
        print(f"\n❌ CRITICAL: Model file not created at {MODEL_PATH}!")
        sys.exit(1)
    print(f"✓ Model trained and saved successfully to {MODEL_PATH}")

    # Step 4: Verify scaler exists
    if SCALER_PATH.exists():
        print(f"✓ Feature scaler saved successfully to {SCALER_PATH}")
    else:
        print("⚠ Feature scaler not found. App will work but with less accuracy.")

    # Step 5: Summary
    print_banner("✅ TRAINING PIPELINE COMPLETED")
    print("📊 Dataset Information:")
    print(f"   • Original dataset: {ORIGINAL_DATASET_PATH}")
    print(f"   • Augmented dataset: {LARGE_DATASET_PATH}")
    print()
    print("🤖 Model Information:")
    print(f"   • Type: VotingClassifier Ensemble")
    print(f"   • Target Accuracy: 80%+")
    print()
    print("📁 Generated Files in artifacts/:")
    print(f"   • {MODEL_PATH.name} - Trained ensemble model")
    print(f"   • {SCALER_PATH.name} - Feature scaler")
    print()


if __name__ == '__main__':
    main()
