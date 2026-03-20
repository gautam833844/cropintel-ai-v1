"""
Master Training & Deployment Script
====================================
This script orchestrates the complete pipeline:
1. Generate 1,000,000 augmented training samples
2. Train the ensemble model to achieve 80%+ accuracy
3. Start the Flask web application
"""

import os
import sys
import subprocess
import time


def print_banner(title):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'─' * 80}")
    print(f"📌 {description}")
    print(f"{'─' * 80}\n")

    try:
        result = subprocess.run(command, shell=True, capture_output=False)
        if result.returncode != 0:
            print(
                f"\n❌ Error: {description} failed with exit code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


def main():
    print_banner("🌾 CROP RECOMMENDATION SYSTEM - MASTER TRAINING PIPELINE")

    # Step 1: Check if original dataset exists
    print("\n[STEP 1] Checking dataset availability...")
    if not os.path.exists('Crop_recommendation.csv'):
        print("❌ ERROR: Crop_recommendation.csv not found!")
        print("   Please ensure the dataset is in the project directory.")
        sys.exit(1)
    print("✓ Original dataset found")

    # Step 2: Generate large dataset
    print("\n[STEP 2] Generating 1,000,000 augmented training samples...")
    if not run_command('python generate_large_dataset.py', 'Dataset generation'):
        print("⚠ Warning: Dataset generation may have had issues. Continuing anyway...")

    # Check if large dataset was created
    if os.path.exists('Crop_recommendation_large.csv'):
        print("✓ Large dataset generated successfully")
    else:
        print("⚠ Large dataset not found. Training script will generate it on-the-fly.")

    # Step 3: Train the ensemble model
    print("\n[STEP 3] Training ensemble model (this may take 5-15 minutes)...")
    if not run_command('python train_model.py', 'Model training'):
        print("\n❌ CRITICAL: Model training failed!")
        print("   Please check the error messages above.")
        sys.exit(1)

    # Check if model was saved
    if not os.path.exists('crop_model.pkl'):
        print("\n❌ CRITICAL: Model file not created!")
        sys.exit(1)
    print("✓ Model trained and saved successfully")

    # Step 4: Verify scaler exists
    if os.path.exists('scaler.pkl'):
        print("✓ Feature scaler saved successfully")
    else:
        print("⚠ Feature scaler not found. App will work but with less accuracy.")

    # Step 5: Summary before Flask startup
    print_banner("✅ TRAINING PIPELINE COMPLETED")
    print("📊 Dataset Information:")
    print(f"   • Original dataset: Crop_recommendation.csv")
    print(f"   • Augmented dataset: Crop_recommendation_large.csv (1,000,000 samples)")
    print()
    print("🤖 Model Information:")
    print(f"   • Type: VotingClassifier Ensemble")
    print(f"   • Components: RandomForest + GradientBoosting + ExtraTrees")
    print(f"   • Target Accuracy: 80%+")
    print()
    print("📁 Generated Files:")
    print(f"   • crop_model.pkl - Trained ensemble model")
    print(f"   • scaler.pkl - Feature scaler")
    print(f"   • rf_model.pkl, gb_model.pkl, et_model.pkl - Individual models")
    print()

    # Step 6: Start Flask application
    print("\n[STEP 4] Starting Flask Web Application...")
    print(f"   URL: http://localhost:5000")
    print(f"   Press Ctrl+C to stop the server\n")

    try:
        subprocess.run('python app.py', shell=True)
    except KeyboardInterrupt:
        print("\n\n✓ Flask application stopped by user")
        print("=" * 80)
        print("Thank you for using the Crop Recommendation System!")
        print("=" * 80)


if __name__ == '__main__':
    main()
