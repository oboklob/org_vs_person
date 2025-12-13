#!/usr/bin/env python
"""Test legal form suffix detection as organization classifier.

Goal: Achieve >93% precision when predicting ORG by checking for legal form suffixes.
This would allow fast pre-filtering before passing to ML classifier.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from name_classifier.iso20275_matcher import ISO20275Matcher

def test_legal_form_detector(tier_filter=None):
    """Test legal form detection as organization predictor.
    
    Args:
        tier_filter: If specified, only use these tiers (e.g., ['A'] or ['A', 'B'])
    
    Returns:
        Dictionary with precision, recall, F1, and counts
    """
    # Load test data
    test_df = pd.read_csv("data/test.csv")
    print(f"Loaded {len(test_df):,} test samples")
    print(f"  ORG: {(test_df['type'] == 'ORG').sum():,}")
    print(f"  PER: {(test_df['type'] == 'PER').sum():,}")
    print()
    
    # Initialize matcher
    matcher = ISO20275Matcher()
    
    # Test predictions
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    tier_stats = {'A': {'tp': 0, 'fp': 0}, 'B': {'tp': 0, 'fp': 0}, 'C': {'tp': 0, 'fp': 0}}
    
    for idx, row in test_df.iterrows():
        name = row['label']
        true_label = row['type']
        
        # Detect legal form
        match = matcher.match_legal_form(name)
        
        # Predict ORG if legal form found (optionally filtered by tier)
        if match:
            if tier_filter is None or match.metadata.tier in tier_filter:
                predicted = 'ORG'
                
                # Track tier stats
                if true_label == 'ORG':
                    tier_stats[match.metadata.tier]['tp'] += 1
                else:
                    tier_stats[match.metadata.tier]['fp'] += 1
            else:
                predicted = None  # Don't predict if tier filtered out
        else:
            predicted = None  # No legal form = no prediction
        
        # Calculate confusion matrix
        if predicted == 'ORG':
            if true_label == 'ORG':
                true_positives += 1
            else:
                false_positives += 1
        else:
            if true_label == 'ORG':
                false_negatives += 1
            else:
                true_negatives += 1
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    coverage = (true_positives + false_positives) / len(test_df)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'coverage': coverage,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'tier_stats': tier_stats
    }

print("=" * 80)
print("LEGAL FORM SUFFIX ORGANIZATION DETECTION")
print("=" * 80)
print()

# Test 1: All tiers
print("Test 1: ALL TIERS (A, B, C)")
print("-" * 80)
results_all = test_legal_form_detector()
print(f"Precision: {results_all['precision']:.4f} {'✅' if results_all['precision'] >= 0.93 else '❌ (need >= 0.93)'}")
print(f"Recall:    {results_all['recall']:.4f}")
print(f"F1 Score:  {results_all['f1']:.4f}")
print(f"Coverage:  {results_all['coverage']:.4f} ({results_all['coverage']*100:.1f}% of test set)")
print(f"")
print(f"Results:")
print(f"  True Positives:  {results_all['true_positives']:,} (correctly identified ORG)")
print(f"  False Positives: {results_all['false_positives']:,} (incorrectly called ORG)")
print(f"  False Negatives: {results_all['false_negatives']:,} (missed ORG)")
print(f"  True Negatives:  {results_all['true_negatives']:,} (correctly no prediction)")
print()
print("By Tier:")
for tier in ['A', 'B', 'C']:
    stats = results_all['tier_stats'][tier]
    tier_precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
    print(f"  Tier {tier}: {stats['tp']:,} TP, {stats['fp']:,} FP → Precision: {tier_precision:.4f}")
print()

# Test 2: Tier A only
print("Test 2: TIER A ONLY (highest confidence)")
print("-" * 80)
results_a = test_legal_form_detector(tier_filter=['A'])
print(f"Precision: {results_a['precision']:.4f} {'✅' if results_a['precision'] >= 0.93 else '❌ (need >= 0.93)'}")
print(f"Recall:    {results_a['recall']:.4f}")
print(f"F1 Score:  {results_a['f1']:.4f}")
print(f"Coverage:  {results_a['coverage']:.4f} ({results_a['coverage']*100:.1f}% of test set)")
print()

# Test 3: Tier A + B
print("Test 3: TIER A + B (exclude ambiguous)")
print("-" * 80)
results_ab = test_legal_form_detector(tier_filter=['A', 'B'])
print(f"Precision: {results_ab['precision']:.4f} {'✅' if results_ab['precision'] >= 0.93 else '❌ (need >= 0.93)'}")
print(f"Recall:    {results_ab['recall']:.4f}")
print(f"F1 Score:  {results_ab['f1']:.4f}")
print(f"Coverage:  {results_ab['coverage']:.4f} ({results_ab['coverage']*100:.1f}% of test set)")
print()

# Summary
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
best = max(
    [('All tiers', results_all), ('Tier A only', results_a), ('Tier A+B', results_ab)],
    key=lambda x: x[1]['precision']
)
print(f"Best precision: {best[0]} → {best[1]['precision']:.4f}")
print()
if best[1]['precision'] >= 0.93:
    print(f"✅ MEETS REQUIREMENT (>= 0.93 precision)")
    print()
    print(f"Recommendation: Use {best[0]} for fast org detection")
    print(f"  - {best[1]['coverage']*100:.1f}% of names can be quickly classified as ORG")
    print(f"  - {(1-best[1]['coverage'])*100:.1f}% need to go through ML classifier")
    print(f"  - {best[1]['precision']*100:.1f}% accuracy when we say it's an ORG")
else:
    print(f"❌ DOES NOT MEET REQUIREMENT")
    print(f"  Best precision achieved: {best[1]['precision']:.4f}")
    print(f"  Need to improve by: {0.93 - best[1]['precision']:.4f}")
