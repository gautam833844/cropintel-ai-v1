"""
Large Dataset Generator for Crop Recommendation
===============================================
Generates 1,000,000 augmented training samples from the original dataset
by creating variations with step size of 0.01 for each feature.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configuration
ORIGINAL_CSV = 'Crop_recommendation.csv'
OUTPUT_CSV = 'Crop_recommendation_large.csv'
# Reduced for faster processing (100k samples = excellent for ML)
TARGET_SAMPLES = 100000
VARIATION_STEP = 0.01

print("=" * 70)
print("LARGE DATASET GENERATOR FOR CROP RECOMMENDATION")
print("=" * 70)

# Load original dataset
print("\n[1/4] Loading original dataset...")
df_original = pd.read_csv(ORIGINAL_CSV)
print(f"✓ Original dataset shape: {df_original.shape}")
print(f"  Crops in dataset: {df_original['label'].unique().tolist()}")

# Get unique crops and their statistics
unique_crops = df_original['label'].unique()
n_crops = len(unique_crops)
samples_per_crop = TARGET_SAMPLES // n_crops
extra_samples = TARGET_SAMPLES % n_crops

print(f"\n[2/4] Generating augmented dataset...")
print(f"  Target samples: {TARGET_SAMPLES:,}")
print(f"  Number of crop types: {n_crops}")
print(f"  Samples per crop: {samples_per_crop:,}")
print(f"  Extra samples: {extra_samples}")

new_data = []

# For each unique crop
for crop_idx, crop in enumerate(unique_crops):
    crop_data = df_original[df_original['label'] == crop]

    # Calculate how many samples needed for this crop
    n_samples = samples_per_crop + (1 if crop_idx < extra_samples else 0)

    # Get the number of features (exclude label)
    n_features = len(crop_data.columns) - 1
    feature_columns = crop_data.columns[:-1].tolist()

    print(f"\n  Processing {crop}...")

    # Generate augmented samples for each original sample
    for orig_idx, orig_row in crop_data.iterrows():
        # Calculate samples to generate from this original row
        samples_from_row = n_samples // len(crop_data)

        for sample_num in range(samples_from_row):
            new_row = {}

            # Add original features with small random variations
            for feature in feature_columns:
                original_value = orig_row[feature]
                # Add random noise between -0.05 and +0.05 with step 0.01
                noise = np.random.choice(
                    np.arange(-0.05, 0.06, VARIATION_STEP))
                new_value = original_value + noise

                # Keep values within reasonable bounds
                new_row[feature] = max(0, new_value)

            # Add crop label
            new_row['label'] = crop
            new_data.append(new_row)

    print(
        f"    ✓ Generated {len([x for x in new_data if x['label'] == crop]):,} samples for {crop}")

# Convert to DataFrame
print(f"\n[3/4] Converting to DataFrame...")
df_augmented = pd.DataFrame(new_data)

# Trim to exact target size
df_augmented = df_augmented.iloc[:TARGET_SAMPLES].copy()

print(f"  ✓ DataFrame created with shape: {df_augmented.shape}")

# Verify distribution
print(f"\n  Crop distribution in augmented dataset:")
crop_counts = df_augmented['label'].value_counts()
for crop, count in crop_counts.items():
    percentage = (count / len(df_augmented)) * 100
    print(f"    {crop}: {count:,} samples ({percentage:.2f}%)")

# Save to CSV
print(f"\n[4/4] Saving augmented dataset...")
df_augmented.to_csv(OUTPUT_CSV, index=False)
print(f"✓ Dataset saved to '{OUTPUT_CSV}'")
print(f"  Total samples: {len(df_augmented):,}")
print(f"  Total features: {len(df_augmented.columns) - 1}")
print(
    f"  File size: {np.round(df_augmented.memory_usage(deep=True).sum() / 1024 / 1024, 2)} MB")

print("\n" + "=" * 70)
print("Dataset generation complete! Ready for model training.")
print("=" * 70)
