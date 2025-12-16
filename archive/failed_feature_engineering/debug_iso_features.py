#!/usr/bin/env python
"""Debug why ISO features are all zero."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from name_classifier import ISO20275Matcher, extract_features

# Load a few samples
df = pd.read_csv("data/train.csv").head(100)

matcher = ISO20275Matcher()

print("Testing ISO feature extraction on sample data:")
print("=" * 80)

zero_count = 0
nonzero_count = 0

for idx, row in df.iterrows():
    name = row['label']
    features = extract_features(name, matcher)
    
    has_legal_form = features[0]
    
    if has_legal_form > 0:
        nonzero_count += 1
        if nonzero_count <= 5:
            print(f"✅ '{name}' → has_legal_form={has_legal_form}")
            print(f"   ISO features: {features[0:8]}")
    else:
        zero_count += 1

print()
print(f"Summary of 100 samples:")
print(f"  Names WITH legal forms: {nonzero_count}")
print(f"  Names WITHOUT legal forms: {zero_count}")
print()

if nonzero_count == 0:
    print("❌ CRITICAL: NO names matched legal forms!")
    print("   This suggests ISO matching is broken or data doesn't have legal forms")
else:
    print(f"✅ ISO matching works ({nonzero_count}% match rate)")
