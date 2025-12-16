#!/usr/bin/env python
"""Check feature ordering in actual training pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
import numpy as np

# Load pipeline
pipeline = joblib.load("name_classifier/models/feature_pipeline.pkl")

# Test on a name with legal form
test_df = pd.DataFrame({'label': ['Acme Corporation Ltd', 'John Smith']})

# Transform
features = pipeline.transform(test_df)

print("Pipeline output shape:", features.shape)
print("Expected: (2, 262176) = 262144 hash + 24 engineered + 8 padding?")
print()

# Check last 24 features (should be engineered features)
print("Last 24 features for 'Acme Corporation Ltd':")
print(features[0, -24:].toarray() if hasattr(features[0, -24:], 'toarray') else features[0, -24:])
print()

print("Last 24 features for 'John Smith':")
print(features[1, -24:].toarray() if hasattr(features[1, -24:], 'toarray') else features[1, -24:])
print()

# Manually extract to compare
from name_classifier import ISO20275Matcher, extract_features
matcher = ISO20275Matcher()

manual_feat1 = extract_features('Acme Corporation Ltd', matcher)
manual_feat2 = extract_features('John Smith', matcher)

print("Manual extraction for 'Acme Corporation Ltd':")
print(manual_feat1)
print()

print("Manual extraction for 'John Smith':")
print(manual_feat2)
print()

print("Do they match the last 24 from pipeline?")
