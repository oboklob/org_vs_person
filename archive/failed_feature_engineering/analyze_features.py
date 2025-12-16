#!/usr/bin/env python
"""Analyze feature importance from trained model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from name_classifier.feature_engineering import get_feature_names

print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Load the trained model
model_path = Path("name_classifier/models/model.pkl")
pipeline_path = Path("name_classifier/models/feature_pipeline.pkl")

if not model_path.exists():
    print(f"Error: Model not found at {model_path}")
    sys.exit(1)

model = joblib.load(model_path)
pipeline = joblib.load(pipeline_path)

print(f"Model type: {type(model).__name__}")
print()

# Check if model has feature importance
if hasattr(model, 'coef_'):
    coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
    
    # Get total feature count
    total_features = len(coef)
    n_engineered = 24  # Updated from 32 - removed 8 language features
    n_hash = total_features - n_engineered
    
    print(f"Total features: {total_features:,}")
    print(f"  - Hash features: {n_hash:,}")
    print(f"  - Engineered features: {n_engineered}")
    print()
    
    # Analyze engineered features (last 32)
    eng_coef = coef[-n_engineered:]
    eng_names = get_feature_names()
    
    print("=" * 80)
    print("ENGINEERED FEATURE COEFFICIENTS")
    print("=" * 80)
    
    # Get class ordering to interpret coefficients correctly
    classes = model.classes_
    print(f"Model classes: {classes}")
    
    if classes[0] == 'ORG' and classes[1] == 'PER':
        print("Positive coefficient → predicts PER (index 1)")
        print("Negative coefficient → predicts ORG (index 0)")
        pos_label, neg_label = "PER", "ORG"
    else:
        print("Positive coefficient → predicts ORG (index 1)")
        print("Negative coefficient → predicts PER (index 0)")
        pos_label, neg_label = "ORG", "PER"
    
    print()
    
    # Sort by absolute coefficient value
    eng_data = list(zip(eng_names, eng_coef))
    eng_data.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("Top 15 Most Important Engineered Features:")
    print("-" * 80)
    for i, (name, coef_val) in enumerate(eng_data[:15], 1):
        direction = f"→ {pos_label}" if coef_val > 0 else f"→ {neg_label}"
        print(f"{i:2d}. {name:35s} {coef_val:8.4f}  {direction}")
    
    print()
    print("Weakest 10 Features (possibly noise):")
    print("-" * 80)
    weak_features = [x for x in eng_data if abs(x[1]) < 0.01]
    for name, coef_val in weak_features[:10]:
        print(f"    {name:35s} {coef_val:8.4f}")
    
    print()
    print("=" * 80)
    print("ANALYSIS BY FEATURE GROUP")
    print("=" * 80)
    
    # Group features (updated for 24 features)
    groups = {
        'ISO Legal Forms (0-7)': eng_data[0:8],
        'String Structure (8-15)': eng_data[8:16],
        'Casing/Initials (16-20)': eng_data[16:21],
        'Lexical Signals (21-23)': eng_data[21:24]
    }
    
    for group_name, features in groups.items():
        if len(features) > 0:
            avg_abs_coef = np.mean([abs(c) for _, c in features])
            max_abs_coef = max([abs(c) for _, c in features])
            print(f"{group_name:30s} Avg|Coef|={avg_abs_coef:.4f}, Max|Coef|={max_abs_coef:.4f}")
        else:
            print(f"{group_name:30s} EMPTY GROUP")
    
    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Check if ISO features are weak
    iso_strength = np.mean([abs(c) for _, c in eng_data[0:8]])
    overall_strength = np.mean([abs(c) for _, c in eng_data])
    
    if iso_strength < overall_strength * 0.5:
        print("⚠️  ISO legal form features are weaker than expected!")
        print("   → Feature dropout may be too aggressive (try --no-dropout)")
        print("   → Or legal forms aren't strong signals in your data")
    
    # Check for very weak features
    if len(weak_features) > 5:
        print(f"⚠️  {len(weak_features)} features have very weak coefficients (< 0.01)")
        print("   → Consider removing these features for faster training")
    
    # Compare hash vs engineered
    hash_strength = np.mean(np.abs(coef[:-n_engineered]))
    eng_strength = np.mean(np.abs(eng_coef))
    
    print()
    print(f"Average coefficient magnitude:")
    print(f"  Hash features:       {hash_strength:.6f}")
    print(f"  Engineered features: {eng_strength:.4f}")
    
    if eng_strength < hash_strength * 10:
        print()
        print("⚠️  Engineered features are weak compared to hash features!")
        print("   → This suggests they may not be adding much value")
        print("   → Try training without engineered features for comparison")
    
    print()
    print("=" * 80)
    print("NEXT STEPS TO IMPROVE ACCURACY")
    print("=" * 80)
    print("1. Try training WITHOUT dropout: --no-dropout")
    print("2. Try training WITHOUT engineered features (use old train_model.py)")
    print("3. Increase n-gram range: try (2, 5) or (3, 6)")
    print("4. Try different regularization: --alpha 0.00001 or 0.001")
    
else:
    print("Model does not have coefficients (not a linear model)")
    print("Cannot analyze feature importance.")

print()
