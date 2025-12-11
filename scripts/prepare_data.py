#!/usr/bin/env python
"""
Data preparation script for the name classifier.

This script:
1. Loads the paranames.tsv.gz dataset
2. Filters to keep only ORG and PER types (removes LOC)
3. Makes dataset unique by label (language-independent)
4. Creates train/test split
5. Optionally samples a subset for experimentation
6. Saves processed data to CSV files
"""
import argparse
import gzip
import pickle
from pathlib import Path
import sys
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split


def load_paranames(corpus_path: Path) -> pd.DataFrame:
    """
    Load the paranames dataset from TSV file.

    Args:
        corpus_path: Path to paranames.tsv.gz file

    Returns:
        DataFrame with columns: wikidata_id, eng, label, language, type
    """
    print(f"Loading paranames dataset from {corpus_path}...")

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    # Read TSV file (gzipped)
    df = pd.read_csv(corpus_path, sep="\t", compression="gzip")

    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"\nType distribution:")
    print(df["type"].value_counts())

    return df


def save_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """
    Save DataFrame to pickle cache.
    
    Args:
        df: DataFrame to save
        cache_path: Path to cache file
    """
    # Create cache directory if it doesn't exist
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with metadata
    cache_data = {
        'data': df,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0',
        'row_count': len(df)
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"\nCache saved to: {cache_path}")
    print(f"Cached {len(df):,} rows")


def load_cache(cache_path: Path) -> pd.DataFrame:
    """
    Load DataFrame from pickle cache.
    
    Args:
        cache_path: Path to cache file
    
    Returns:
        Cached DataFrame or None if cache doesn't exist/is invalid
    """
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"\n{'='*60}")
        print("Loading from cache...")
        print('='*60)
        print(f"Cache file: {cache_path}")
        print(f"Cached on: {cache_data['timestamp']}")
        print(f"Cache version: {cache_data['version']}")
        print(f"Rows in cache: {cache_data['row_count']:,}")
        
        return cache_data['data']
    except Exception as e:
        print(f"\nWarning: Failed to load cache: {e}")
        print("Will rebuild from source data...")
        return None


def filter_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to keep only ORG and PER, then deduplicate by label.

    Args:
        df: Input DataFrame

    Returns:
        Filtered and deduplicated DataFrame
    """
    print("\n" + "=" * 60)
    print("Filtering and deduplicating data...")
    print("=" * 60)

    # Filter: keep only ORG and PER
    print(f"\nOriginal size: {len(df):,} rows")
    df_filtered = df[df["type"].isin(["ORG", "PER"])].copy()
    print(f"After filtering (ORG, PER only): {len(df_filtered):,} rows")

    print(f"\nType distribution after filtering:")
    print(df_filtered["type"].value_counts())

    # Deduplicate by label (keep first occurrence)
    # This makes the dataset language-independent
    print(f"\nDeduplicating by label...")
    print(f"Before deduplication: {len(df_filtered):,} rows")
    print(f"Unique labels: {df_filtered['label'].nunique():,}")

    df_dedup = df_filtered.drop_duplicates(subset=["label"], keep="first")

    print(f"After deduplication: {len(df_dedup):,} rows")
    print(f"\nFinal type distribution:")
    print(df_dedup["type"].value_counts())

    return df_dedup


def create_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/test split.

    Args:
        df: Input DataFrame
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    print("\n" + "=" * 60)
    print("Creating train/test split...")
    print("=" * 60)

    # Use only label and type columns
    df_clean = df[["label", "type"]].copy()

    # Stratified split to maintain class balance
    train_df, test_df = train_test_split(
        df_clean, test_size=test_size, stratify=df_clean["type"], random_state=random_state
    )

    print(f"\nTrain size: {len(train_df):,} rows")
    print(f"Train type distribution:")
    print(train_df["type"].value_counts())

    print(f"\nTest size: {len(test_df):,} rows")
    print(f"Test type distribution:")
    print(test_df["type"].value_counts())

    return train_df, test_df


def sample_data(df: pd.DataFrame, sample_size: int, random_state: int = 42) -> pd.DataFrame:
    """
    Sample a subset of the data, maintaining class balance.

    Args:
        df: Input DataFrame
        sample_size: Number of samples to take
        random_state: Random seed

    Returns:
        Sampled DataFrame
    """
    if sample_size >= len(df):
        print(f"\nSample size ({sample_size:,}) >= dataset size ({len(df):,}), using full dataset")
        return df

    print(f"\nSampling {sample_size:,} rows (stratified by type)...")

    # Calculate samples per class to maintain balance
    type_counts = df["type"].value_counts()
    n_per_class = {t: int(sample_size * (count / len(df))) for t, count in type_counts.items()}

    # Adjust to ensure we hit the target sample size
    total = sum(n_per_class.values())
    if total < sample_size:
        # Add remaining samples to the majority class
        majority_class = type_counts.index[0]
        n_per_class[majority_class] += sample_size - total

    # Sample from each class
    sampled_dfs = []
    for type_name, n in n_per_class.items():
        class_df = df[df["type"] == type_name]
        if n > len(class_df):
            n = len(class_df)
        sampled_dfs.append(class_df.sample(n=n, random_state=random_state))

    df_sampled = pd.concat(sampled_dfs, ignore_index=True)

    # Shuffle
    df_sampled = df_sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Sampled {len(df_sampled):,} rows")
    print(f"Type distribution in sample:")
    print(df_sampled["type"].value_counts())

    return df_sampled


def main():
    parser = argparse.ArgumentParser(description="Prepare data for name classifier training")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("corpus/paranames.tsv.gz"),
        help="Path to paranames.tsv.gz file (default: corpus/paranames.tsv.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for processed data (default: data/)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional: sample N rows for experimentation (default: use full dataset)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set proportion (default: 0.2)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Use pickle cache for deduped data (default: True)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        help="Disable pickle cache",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/cache/deduped_data.pkl"),
        help="Path to pickle cache file (default: data/cache/deduped_data.pkl)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild from source, ignoring any existing cache",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for processing (default: -1 for all cores)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load from cache if enabled
    df_processed = None
    if args.use_cache and not args.force_rebuild:
        df_processed = load_cache(args.cache_path)
    
    # If no cache or cache disabled, load and process from source
    if df_processed is None:
        # Load data
        df = load_paranames(args.corpus_path)

        # Filter and deduplicate
        df_processed = filter_and_deduplicate(df)
        
        # Save to cache if enabled
        if args.use_cache:
            save_cache(df_processed, args.cache_path)

    # Sample if requested
    if args.sample_size:
        df_processed = sample_data(df_processed, args.sample_size, args.random_state)

    # Create train/test split
    train_df, test_df = create_train_test_split(
        df_processed, test_size=args.test_size, random_state=args.random_state
    )

    # Save to CSV
    train_path = args.output_dir / "train.csv"
    test_path = args.output_dir / "test.csv"

    print("\n" + "=" * 60)
    print("Saving processed data...")
    print("=" * 60)

    train_df.to_csv(train_path, index=False)
    print(f"\nTrain data saved to: {train_path}")

    test_df.to_csv(test_path, index=False)
    print(f"Test data saved to: {test_path}")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Train models: python scripts/train_model.py")
    print(f"  2. Run tests: pytest")


if __name__ == "__main__":
    main()
