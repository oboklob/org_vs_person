#!/usr/bin/env python3
"""Benchmark script to assess fast filter vs model-only performance.

This script compares the performance of:
1. Fast filter + model (filter catches some orgs, model handles the rest)
2. Model-only (all names classified by ML model)

The goal is to determine if the fast filter provides a meaningful speed advantage.
"""
import time
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


def benchmark_model_only(classifier: NameClassifier, names: List[str]) -> Tuple[float, List[str]]:
    """Benchmark model-only classification.
    
    Args:
        classifier: NameClassifier instance
        names: List of names to classify
        
    Returns:
        Tuple of (elapsed_time_seconds, predictions)
    """
    start = time.perf_counter()
    predictions = classifier.classify_list(names)
    elapsed = time.perf_counter() - start
    
    return elapsed, predictions


def benchmark_filter_then_model(
    detector: FastOrgDetector,
    classifier: NameClassifier,
    names: List[str]
) -> Tuple[float, List[str], Dict]:
    """Benchmark fast filter + model classification.
    
    Args:
        detector: FastOrgDetector instance
        classifier: NameClassifier instance
        names: List of names to classify
        
    Returns:
        Tuple of (elapsed_time_seconds, predictions, stats)
    """
    start = time.perf_counter()
    
    # Track statistics
    stats = {
        'filter_time': 0.0,
        'model_time': 0.0,
        'filtered_count': 0,
        'model_count': 0
    }
    
    # Phase 1: Fast filter
    filter_start = time.perf_counter()
    detected_orgs, needs_classification = detector.filter_orgs(names)
    stats['filter_time'] = time.perf_counter() - filter_start
    stats['filtered_count'] = len(detected_orgs)
    stats['model_count'] = len(needs_classification)
    
    # Phase 2: Model classification for remaining names
    model_start = time.perf_counter()
    if needs_classification:
        model_predictions = classifier.classify_list(needs_classification)
    else:
        model_predictions = []
    stats['model_time'] = time.perf_counter() - model_start
    
    # Reconstruct full predictions in original order
    predictions = []
    org_idx = 0
    model_idx = 0
    
    for name in names:
        if name in detected_orgs:
            predictions.append('ORG')
            org_idx += 1
        else:
            predictions.append(model_predictions[model_idx])
            model_idx += 1
    
    elapsed = time.perf_counter() - start
    
    return elapsed, predictions, stats


def run_benchmark(
    sample_sizes: List[int],
    tier_filter: List[str] = None,
    num_runs: int = 3
) -> pd.DataFrame:
    """Run comprehensive benchmark across different sample sizes.
    
    Args:
        sample_sizes: List of sample sizes to test
        tier_filter: Tier filter for FastOrgDetector
        num_runs: Number of runs for each size (for averaging)
        
    Returns:
        DataFrame with benchmark results
    """
    print("\n" + "="*80)
    print(f"BENCHMARK: Fast Filter vs Model-Only Performance")
    print(f"Tier filter: {tier_filter or 'All (A, B, C)'}")
    print(f"Runs per size: {num_runs}")
    print("="*80)
    
    # Initialize components
    classifier = NameClassifier()
    detector = FastOrgDetector(tier_filter=tier_filter)
    
    # Load full test dataset
    print("\nLoading test data...")
    test_df = load_test_data()
    print(f"Loaded {len(test_df)} test samples")
    
    results = []
    
    for size in sample_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {size:,} samples")
        print(f"{'='*60}")
        
        # Get sample
        if size > len(test_df):
            print(f"  Warning: Requested {size} samples but only {len(test_df)} available")
            size = len(test_df)
        
        sample_df = test_df.head(size)
        names = sample_df['name'].tolist()
        
        # Run multiple times for stable measurements
        model_times = []
        filter_times = []
        filter_stats_list = []
        
        for run in range(num_runs):
            print(f"\n  Run {run + 1}/{num_runs}:")
            
            # Benchmark model-only
            model_time, model_preds = benchmark_model_only(classifier, names)
            model_times.append(model_time)
            print(f"    Model-only: {model_time:.4f}s")
            
            # Benchmark filter + model
            filter_time, filter_preds, stats = benchmark_filter_then_model(
                detector, classifier, names
            )
            filter_times.append(filter_time)
            filter_stats_list.append(stats)
            print(f"    Filter + Model: {filter_time:.4f}s")
            print(f"      - Filter caught: {stats['filtered_count']:,} ({stats['filtered_count']/size*100:.1f}%)")
            print(f"      - Model needed: {stats['model_count']:,} ({stats['model_count']/size*100:.1f}%)")
            print(f"      - Filter time: {stats['filter_time']:.4f}s")
            print(f"      - Model time: {stats['model_time']:.4f}s")
        
        # Calculate averages
        avg_model_time = np.mean(model_times)
        avg_filter_time = np.mean(filter_times)
        avg_filter_caught = np.mean([s['filtered_count'] for s in filter_stats_list])
        avg_filter_phase_time = np.mean([s['filter_time'] for s in filter_stats_list])
        avg_model_phase_time = np.mean([s['model_time'] for s in filter_stats_list])
        
        speedup = (avg_model_time / avg_filter_time - 1) * 100
        
        print(f"\n  Average results:")
        print(f"    Model-only: {avg_model_time:.4f}s")
        print(f"    Filter + Model: {avg_filter_time:.4f}s")
        print(f"    Speedup: {speedup:+.1f}% {'(faster)' if speedup > 0 else '(slower)'}")
        print(f"    Throughput (model-only): {size/avg_model_time:,.0f} names/sec")
        print(f"    Throughput (filter+model): {size/avg_filter_time:,.0f} names/sec")
        
        results.append({
            'sample_size': size,
            'model_only_time': avg_model_time,
            'filter_model_time': avg_filter_time,
            'speedup_pct': speedup,
            'filter_coverage_pct': avg_filter_caught / size * 100,
            'filter_phase_time': avg_filter_phase_time,
            'model_phase_time': avg_model_phase_time,
            'model_only_throughput': size / avg_model_time,
            'filter_model_throughput': size / avg_filter_time,
        })
    
    return pd.DataFrame(results)


def print_summary(results_df: pd.DataFrame):
    """Print summary of benchmark results.
    
    Args:
        results_df: DataFrame with benchmark results
    """
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Overall statistics
    avg_speedup = results_df['speedup_pct'].mean()
    avg_coverage = results_df['filter_coverage_pct'].mean()
    
    print(f"\nAverage speedup from filter: {avg_speedup:+.1f}%")
    print(f"Average filter coverage: {avg_coverage:.1f}%")
    
    # Show when filter is beneficial
    beneficial = results_df[results_df['speedup_pct'] > 0]
    if len(beneficial) > 0:
        print(f"\nFilter provides speedup in {len(beneficial)}/{len(results_df)} test sizes")
        print(f"Best speedup: {results_df['speedup_pct'].max():+.1f}% at {results_df.loc[results_df['speedup_pct'].idxmax(), 'sample_size']:,.0f} samples")
    else:
        print(f"\nFilter does NOT provide speedup in any test size")
    
    # Time breakdown
    print("\nTime breakdown (average across all sizes):")
    print(f"  Filter phase: {results_df['filter_phase_time'].mean():.4f}s")
    print(f"  Model phase (in filter+model): {results_df['model_phase_time'].mean():.4f}s")
    print(f"  Model phase (in model-only): {results_df['model_only_time'].mean():.4f}s")
    
    # Detailed table
    print("\nDetailed Results:")
    print(results_df.to_string(index=False))
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    if avg_speedup > 5:
        print(f"✓ The fast filter provides a meaningful speedup ({avg_speedup:.1f}% on average).")
        print(f"  RECOMMEND: Keep the filter for performance optimization.")
    elif avg_speedup > 0:
        print(f"~ The fast filter provides marginal speedup ({avg_speedup:.1f}% on average).")
        print(f"  CONSIDER: Keep if simplicity is less important than performance.")
    else:
        print(f"✗ The fast filter provides NO speedup (actually {avg_speedup:.1f}% slower on average).")
        print(f"  RECOMMEND: Remove the filter to simplify the codebase.")
        print(f"  The ML model is fast enough that pre-filtering adds overhead without benefit.")
    print("="*80)


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Benchmark fast filter vs model-only classification performance"
    )
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=[100, 1000, 10000, 50000],
        help='Sample sizes to test (default: 100 1000 10000 50000)'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=3,
        help='Number of runs per size for averaging (default: 3)'
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
        '--output',
        type=str,
        default=None,
        help='Optional CSV file to save results'
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    results_df = run_benchmark(
        sample_sizes=args.sizes,
        tier_filter=args.tier_filter,
        num_runs=args.runs
    )
    
    # Print summary
    print_summary(results_df)
    
    # Save results if requested
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
