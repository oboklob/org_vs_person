#!/usr/bin/env python
"""Debug feature extraction to find why features are inverted."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from name_classifier import ISO20275Matcher, extract_features

print("=" * 80)
print("DEBUGGING FEATURE EXTRACTION")
print("=" * 80)

# Test with obvious examples
test_cases = [
    ("IBM", "ORG", "All caps acronym"),
    ("Google Inc", "ORG", "Has org keyword 'inc'"),
    ("John Smith", "PER", "Common person name"),
    ("Acme Corporation Ltd", "ORG", "Has legal form 'ltd'"),
    ("Dr. Jane Doe", "PER", "Has person title"),
]

matcher = ISO20275Matcher()

print("\nTesting feature extraction on obvious examples:")
print("-" * 80)

for name, expected_label, description in test_cases:
    print(f"\nName: '{name}' (Expected: {expected_label})")
    print(f"Description: {description}")
    
    features = extract_features(name, matcher)
    feature_names = [
        'has_legal_form', 'legal_form_tier_a', 'legal_form_tier_b', 'legal_form_tier_c',
        'legal_form_token_len', 'legal_form_char_len', 'legal_form_is_ambiguous_short',
        'legal_form_country_is_known', 'has_language_hint', 'has_jurisdiction_hint',
        'language_bucket_en_like', 'language_bucket_germanic', 'language_bucket_romance',
        'language_bucket_nordic', 'language_bucket_other', 'jurisdiction_reserved',
        'char_len', 'token_count', 'avg_token_len', 'contains_digit',
        'contains_ampersand', 'contains_comma', 'contains_slash_or_hyphen', 'contains_period',
        'all_caps_ratio', 'capitalised_token_ratio', 'has_initials_pattern',
        'has_person_title', 'has_person_suffix', 'has_org_keyword', 'has_given_name_hit',
        'has_surname_particle'
    ]
    
    print("Non-zero features:")
    for i, (feat_name, feat_val) in enumerate(zip(feature_names, features)):
        if feat_val != 0:
            print(f"  {feat_name}: {feat_val}")

print("\n" + "=" * 80)
print("CHECKING SPECIFIC FEATURES")
print("=" * 80)

# Test IBM - should have all_caps_ratio = 1.0
ibm_features = extract_features("IBM", matcher)
print(f"\nIBM all_caps_ratio: {ibm_features[24]}")
print(f"Expected: 1.0 (all characters are capitals)")

# Test with org keyword
inc_features = extract_features("Google Inc", matcher)
print(f"\nGoogle Inc has_org_keyword: {inc_features[29]}")
print(f"Expected: 1.0 ('inc' is an org keyword)")

# Test with given name
john_features = extract_features("John Smith", matcher)
print(f"\nJohn Smith has_given_name_hit: {john_features[30]}")
print(f"Expected: 1.0 ('john' is a given name)")

# Test with legal form
ltd_features = extract_features("Company Ltd", matcher)
print(f"\nCompany Ltd has_legal_form: {ltd_features[0]}")
print(f"Expected: 1.0 ('ltd' is a legal form)")

print("\n" + "=" * 80)
print("TESTING PIPELINE EXTRACTION")
print("=" * 80)

# Test the actual pipeline
from name_classifier.transformers import NameFeatureExtractor
import pandas as pd

extractor = NameFeatureExtractor()
test_df = pd.DataFrame({'label': ['IBM', 'Google Inc', 'John Smith']})

extractor.fit(test_df)
pipeline_features = extractor.transform(test_df)

print("\nPipeline extracted features shape:", pipeline_features.shape)
print("\nIBM features from pipeline:")
print("  all_caps_ratio (index 24):", pipeline_features[0, 24])
print("\nGoogle Inc features from pipeline:")
print("  has_org_keyword (index 29):", pipeline_features[1, 29])
print("\nJohn Smith features from pipeline:")
print("  has_given_name_hit (index 30):", pipeline_features[2, 30])

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if pipeline_features[0, 24] == 1.0:
    print("✅ all_caps_ratio is correctly extracted (IBM = 1.0)")
else:
    print(f"❌ all_caps_ratio is WRONG (IBM = {pipeline_features[0, 24]}, expected 1.0)")

if pipeline_features[1, 29] == 1.0:
    print("✅ has_org_keyword is correctly extracted (Google Inc = 1.0)")
else:
    print(f"❌ has_org_keyword is WRONG (Google Inc = {pipeline_features[1, 29]}, expected 1.0)")

if pipeline_features[2, 30] == 1.0:
    print("✅ has_given_name_hit is correctly extracted (John Smith = 1.0)")
else:
    print(f"❌ has_given_name_hit is WRONG (John Smith = {pipeline_features[2, 30]}, expected 1.0)")

print("\n" + "=" * 80)
print("HYPOTHESIS")
print("=" * 80)
print("If features are extracted correctly but coefficients are inverted,")
print("the issue may be:")
print("1. Labels are swapped (ORG/PER reversed in training data)")
print("2. Feature Union is concatenating in wrong order")
print("3. Model classes_ are in unexpected order")
