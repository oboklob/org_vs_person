#!/usr/bin/env python3
"""Benchmark script to assess ISO20275 filter accuracy vs ML classifier accuracy.

This script compares the accuracy of:
1. ISO20275 fast filter (when it detects ORG)
2. ML classifier (overall and on same cases)

The goal is to determine if the filter provides higher precision than the model.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

from org_vs_person.classifier import NameClassifier
from org_vs_person.fast_org_detector import FastOrgDetector
from org_vs_person.config import DATA_DIR


def load_test_data(limit: int = None) -> pd.DataFrame:
    """Load test data from dataset.
    
    Args:
        limit: Optional limit on number of samples to load
        
    Returns:
        DataFrame with 'name' and 'label' columns (renamed from 'label' and 'type')
    """
    test_path = DATA_DIR / "test.csv"
    
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}."
        )
    
    # Columns are: label, type
    # We need to rename: label -> name, type -> label
    df = pd.read_csv(test_path)
    df = df.rename(columns={'label': 'name', 'type': 'label'})
    
    if limit:
        df = df.head(limit)
    
    return df


def evaluate_filter_accuracy(
    detector: FastOrgDetector,
    names: List[str],
    true_labels: List[str]
) -> Dict:
    """Evaluate filter accuracy on test data.
    
    Args:
        detector: FastOrgDetector instance
        names: List of names to classify
        true_labels: Ground truth labels
        
    Returns:
        Dictionary with accuracy metrics
    """
    results = detector.detect_batch(names)
    
    # Track metrics
    filter_detected_org = []
    filter_detected_indices = []
    
    for i, result in enumerate(results):
        if result.is_org:
            filter_detected_org.append(i)
            filter_detected_indices.append(i)
    
    # Calculate metrics for filter detections
    total_detected = len(filter_detected_org)
    
    if total_detected == 0:
        return {
            'total_samples': len(names),
            'filter_detected': 0,
            'filter_coverage': 0.0,
            'filter_precision': 0.0,
            'filter_recall': 0.0,
            'filter_f1': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
        }
    
    # Count true positives and false positives
    true_positives = sum(1 for i in filter_detected_org if true_labels[i] == 'ORG')
    false_positives = total_detected - true_positives
    
    # Count false negatives (ORGs that filter missed)
    total_orgs = sum(1 for label in true_labels if label == 'ORG')
    false_negatives = total_orgs - true_positives
    
    # Calculate metrics
    precision = true_positives / total_detected if total_detected > 0 else 0.0
    recall = true_positives / total_orgs if total_orgs > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'total_samples': len(names),
        'total_orgs': total_orgs,
        'filter_detected': total_detected,
        'filter_coverage': total_detected / len(names) * 100,
        'filter_precision': precision,
        'filter_recall': recall,
        'filter_f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'filter_detected_indices': filter_detected_indices,
    }


def evaluate_classifier_accuracy(
    classifier: NameClassifier,
    names: List[str],
    true_labels: List[str]
) -> Dict:
    """Evaluate classifier accuracy on test data.
    
    Args:
        classifier: NameClassifier instance
        names: List of names to classify
        true_labels: Ground truth labels
        
    Returns:
        Dictionary with accuracy metrics
    """
    predictions = classifier.classify_list(names)
    
    # Calculate metrics
    correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    total = len(names)
    accuracy = correct / total
    
    # Calculate per-class metrics
    org_predictions = [i for i, pred in enumerate(predictions) if pred == 'ORG']
    per_predictions = [i for i, pred in enumerate(predictions) if pred == 'PER']
    
    org_true = sum(1 for label in true_labels if label == 'ORG')
    per_true = sum(1 for label in true_labels if label == 'PER')
    
    # ORG metrics
    org_tp = sum(1 for i in org_predictions if true_labels[i] == 'ORG')
    org_fp = len(org_predictions) - org_tp
    org_fn = org_true - org_tp
    
    org_precision = org_tp / len(org_predictions) if org_predictions else 0.0
    org_recall = org_tp / org_true if org_true > 0 else 0.0
    org_f1 = 2 * org_precision * org_recall / (org_precision + org_recall) if (org_precision + org_recall) > 0 else 0.0
    
    # PER metrics
    per_tp = sum(1 for i in per_predictions if true_labels[i] == 'PER')
    per_fp = len(per_predictions) - per_tp
    per_fn = per_true - per_tp
    
    per_precision = per_tp / len(per_predictions) if per_predictions else 0.0
    per_recall = per_tp / per_true if per_true > 0 else 0.0
    per_f1 = 2 * per_precision * per_recall / (per_precision + per_recall) if (per_precision + per_recall) > 0 else 0.0
    
    return {
        'total_samples': total,
        'accuracy': accuracy,
        'org_precision': org_precision,
        'org_recall': org_recall,
        'org_f1': org_f1,
        'org_tp': org_tp,
        'org_fp': org_fp,
        'org_fn': org_fn,
        'per_precision': per_precision,
        'per_recall': per_recall,
        'per_f1': per_f1,
        'per_tp': per_tp,
        'per_fp': per_fp,
        'per_fn': per_fn,
    }


def compare_on_filter_detections(
    classifier: NameClassifier,
    names: List[str],
    true_labels: List[str],
    filter_detected_indices: List[int]
) -> Dict:
    """Compare classifier accuracy on cases where filter detected ORG.
    
    Args:
        classifier: NameClassifier instance
        names: List of all names
        true_labels: Ground truth labels
        filter_detected_indices: Indices where filter detected ORG
        
    Returns:
        Dictionary with comparison metrics
    """
    if not filter_detected_indices:
        return {
            'total_compared': 0,
            'classifier_agrees': 0,
            'agreement_rate': 0.0,
            'classifier_correct': 0,
            'classifier_accuracy_on_filter': 0.0,
        }
    
    # Get classifier predictions for filter-detected cases
    filter_names = [names[i] for i in filter_detected_indices]
    filter_true = [true_labels[i] for i in filter_detected_indices]
    filter_preds = classifier.classify_list(filter_names)
    
    # Calculate agreement
    agrees = sum(1 for pred in filter_preds if pred == 'ORG')
    agreement_rate = agrees / len(filter_detected_indices)
    
    # Calculate classifier accuracy on these cases
    correct = sum(1 for pred, true in zip(filter_preds, filter_true) if pred == true)
    accuracy_on_filter = correct / len(filter_detected_indices)
    
    return {
        'total_compared': len(filter_detected_indices),
        'classifier_agrees': agrees,
        'agreement_rate': agreement_rate,
        'classifier_correct': correct,
        'classifier_accuracy_on_filter': accuracy_on_filter,
    }


def run_accuracy_benchmark(
    sample_size: int = 50000,
    tier_filter: List[str] = None
) -> Dict:
    """Run comprehensive accuracy benchmark.
    
    Args:
        sample_size: Number of samples to test
        tier_filter: Tier filter for FastOrgDetector
        
    Returns:
        Dictionary with all benchmark results
    """
    print("\n" + "="*80)
    print(f"ACCURACY BENCHMARK: ISO20275 Filter vs ML Classifier")
    print(f"Sample size: {sample_size:,}")
    print(f"Tier filter: {tier_filter or 'All (A, B, C)'}")
    print("="*80)
    
    # Initialize components
    classifier = NameClassifier()
    detector = FastOrgDetector(tier_filter=tier_filter)
    
    # Load test data
    print("\nLoading test data...")
    test_df = load_test_data(limit=sample_size)
    print(f"Loaded {len(test_df)} test samples")
    
    names = test_df['name'].tolist()
    true_labels = test_df['label'].tolist()
    
    # Evaluate filter
    print("\n" + "-"*80)
    print("Evaluating ISO20275 Filter...")
    print("-"*80)
    filter_metrics = evaluate_filter_accuracy(detector, names, true_labels)
    
    print(f"\nFilter Coverage: {filter_metrics['filter_coverage']:.1f}% ({filter_metrics['filter_detected']:,}/{filter_metrics['total_samples']:,})")
    print(f"Filter Precision: {filter_metrics['filter_precision']:.4f} ({filter_metrics['filter_precision']*100:.2f}%)")
    print(f"Filter Recall: {filter_metrics['filter_recall']:.4f} ({filter_metrics['filter_recall']*100:.2f}%)")
    print(f"Filter F1-Score: {filter_metrics['filter_f1']:.4f}")
    print(f"\nTrue Positives: {filter_metrics['true_positives']:,}")
    print(f"False Positives: {filter_metrics['false_positives']:,}")
    print(f"False Negatives: {filter_metrics['false_negatives']:,}")
    
    # Evaluate classifier
    print("\n" + "-"*80)
    print("Evaluating ML Classifier...")
    print("-"*80)
    classifier_metrics = evaluate_classifier_accuracy(classifier, names, true_labels)
    
    print(f"\nOverall Accuracy: {classifier_metrics['accuracy']:.4f} ({classifier_metrics['accuracy']*100:.2f}%)")
    print(f"\nORG Class:")
    print(f"  Precision: {classifier_metrics['org_precision']:.4f} ({classifier_metrics['org_precision']*100:.2f}%)")
    print(f"  Recall: {classifier_metrics['org_recall']:.4f} ({classifier_metrics['org_recall']*100:.2f}%)")
    print(f"  F1-Score: {classifier_metrics['org_f1']:.4f}")
    print(f"PER Class:")
    print(f"  Precision: {classifier_metrics['per_precision']:.4f} ({classifier_metrics['per_precision']*100:.2f}%)")
    print(f"  Recall: {classifier_metrics['per_recall']:.4f} ({classifier_metrics['per_recall']*100:.2f}%)")
    print(f"  F1-Score: {classifier_metrics['per_f1']:.4f}")
    
    # Compare on filter detections
    print("\n" + "-"*80)
    print("Comparing Classifier on Filter-Detected Cases...")
    print("-"*80)
    comparison = compare_on_filter_detections(
        classifier, names, true_labels, 
        filter_metrics['filter_detected_indices']
    )
    
    print(f"\nCases where filter detected ORG: {comparison['total_compared']:,}")
    print(f"Classifier agrees (also says ORG): {comparison['classifier_agrees']:,} ({comparison['agreement_rate']*100:.1f}%)")
    print(f"Classifier correct on these cases: {comparison['classifier_correct']:,} ({comparison['classifier_accuracy_on_filter']*100:.2f}%)")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print(f"\nPrecision (when detecting ORG):")
    print(f"  Filter: {filter_metrics['filter_precision']*100:.2f}%")
    print(f"  Classifier: {classifier_metrics['org_precision']*100:.2f}%")
    
    if filter_metrics['filter_precision'] > classifier_metrics['org_precision']:
        diff = (filter_metrics['filter_precision'] - classifier_metrics['org_precision']) * 100
        print(f"  → Filter is {diff:.2f}pp more precise")
    else:
        diff = (classifier_metrics['org_precision'] - filter_metrics['filter_precision']) * 100
        print(f"  → Classifier is {diff:.2f}pp more precise")
    
    print(f"\nRecall (coverage of true ORGs):")
    print(f"  Filter: {filter_metrics['filter_recall']*100:.2f}%")
    print(f"  Classifier: {classifier_metrics['org_recall']*100:.2f}%")
    
    print(f"\nF1-Score (harmonic mean):")
    print(f"  Filter: {filter_metrics['filter_f1']:.4f}")
    print(f"  Classifier: {classifier_metrics['org_f1']:.4f}")
    
    # Return all metrics
    return {
        'filter': filter_metrics,
        'classifier': classifier_metrics,
        'comparison': comparison,
    }


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Benchmark ISO20275 filter vs classifier accuracy"
    )
    parser.add_argument(
        '--size',
        type=int,
        default=50000,
        help='Sample size to test (default: 50000)'
    )
    parser.add_argument(
        '--tier-filter',
        type=str,
        nargs='+',
        choices=['A', 'B', 'C'],
        default=None,
        help='Tier filter for fast detector (default: all tiers). Example: --tier-filter A B'
    )
    parser.add_argument(
        '--compare-tiers',
        action='store_true',
        help='Compare accuracy across different tier filters'
    )
    
    args = parser.parse_args()
    
    if args.compare_tiers:
        # Compare across tier configurations
        tier_configs = [
            (['A'], "Tier A only (most conservative)"),
            (['A', 'B'], "Tier A+B (recommended)"),
            (None, "All tiers (A, B, C)"),
        ]
        
        print("\n" + "="*80)
        print("COMPARING TIER CONFIGURATIONS")
        print("="*80)
        
        results = []
        for tier_filter, description in tier_configs:
            print(f"\n{'='*80}")
            print(f"{description}")
            print(f"{'='*80}")
            result = run_accuracy_benchmark(
                sample_size=args.size,
                tier_filter=tier_filter
            )
            results.append((description, result))
        
        # Summary comparison
        print("\n" + "="*80)
        print("TIER CONFIGURATION COMPARISON")
        print("="*80)
        print(f"\n{'Config':<35} {'Coverage':<12} {'Precision':<12} {'Recall':<12} {'F1':<10}")
        print("-"*80)
        for desc, result in results:
            f_metrics = result['filter']
            print(f"{desc:<35} {f_metrics['filter_coverage']:>10.1f}% {f_metrics['filter_precision']:>10.2%} {f_metrics['filter_recall']:>10.2%} {f_metrics['filter_f1']:>10.4f}")
        
    else:
        # Single configuration
        run_accuracy_benchmark(
            sample_size=args.size,
            tier_filter=args.tier_filter
        )


if __name__ == "__main__":
    main()
