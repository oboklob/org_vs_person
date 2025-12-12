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
import hashlib
import pickle
from pathlib import Path
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
import yaml
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


def load_corpus_config(config_path: Path) -> Optional[dict]:
    """
    Load and validate YAML corpus configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary or None if file doesn't exist

    Raises:
        ValueError: If configuration is invalid
    """
    if not config_path.exists():
        return None

    print(f"Loading corpus configuration from {config_path}...")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}")

    # Validate configuration structure
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")

    if 'sources' not in config:
        raise ValueError("Configuration must have a 'sources' key")

    if not isinstance(config['sources'], list):
        raise ValueError("'sources' must be a list")

    if len(config['sources']) == 0:
        raise ValueError("At least one source must be configured")

    # Validate each source
    for i, source in enumerate(config['sources']):
        required_keys = ['name', 'path', 'format', 'columns']
        for key in required_keys:
            if key not in source:
                raise ValueError(f"Source {i} missing required key: {key}")

        # Validate columns
        if 'name' not in source['columns']:
            raise ValueError(f"Source {i} columns must specify 'name'")

        # Must have either classification column or fixed_classification
        has_classification = source['columns'].get('classification') is not None
        has_fixed = 'fixed_classification' in source

        if not has_classification and not has_fixed:
            raise ValueError(
                f"Source {i} ({source['name']}) must have either "
                f"columns.classification or fixed_classification"
            )

    print(f"Configuration loaded successfully with {len(config['sources'])} source(s)")
    return config


def get_config_hash(config_path: Path) -> str:
    """
    Compute hash of configuration file for cache invalidation.

    Args:
        config_path: Path to configuration file

    Returns:
        MD5 hash of file contents
    """
    if not config_path.exists():
        return "no_config"

    with open(config_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def load_corpus_source(source_config: dict) -> pd.DataFrame:
    """
    Load a single corpus source based on its configuration.

    Args:
        source_config: Configuration dictionary for a single source

    Returns:
        DataFrame with normalized columns: ['label', 'type']
    """
    name = source_config['name']
    path = Path(source_config['path'])
    file_format = source_config['format']
    compression = source_config.get('compression')
    columns = source_config['columns']
    fixed_classification = source_config.get('fixed_classification')
    filters = source_config.get('filters', {})

    print(f"\nLoading source: {name}")
    print(f"  Path: {path}")
    print(f"  Format: {file_format}")
    print(f"  Compression: {compression}")

    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    # Load file based on format
    if file_format == 'tsv':
        df = pd.read_csv(path, sep='\t', compression=compression)
    elif file_format == 'csv':
        df = pd.read_csv(path, compression=compression)
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    print(f"  Loaded {len(df):,} rows")

    # Extract and rename columns
    name_col = columns['name']
    classification_col = columns.get('classification')

    if name_col not in df.columns:
        raise ValueError(f"Column '{name_col}' not found in {path}")

    # Create normalized dataframe
    result_df = pd.DataFrame()
    result_df['label'] = df[name_col]

    if classification_col:
        # Use classification from column
        if classification_col not in df.columns:
            raise ValueError(f"Column '{classification_col}' not found in {path}")
        result_df['type'] = df[classification_col]
    elif fixed_classification:
        # Use fixed classification for all rows
        result_df['type'] = fixed_classification
        print(f"  Applied fixed classification: {fixed_classification}")
    else:
        raise ValueError(f"Source {name} has no classification method")

    # Apply filters
    if 'type' in filters:
        types_to_keep = filters['type']
        before_count = len(result_df)
        result_df = result_df[result_df['type'].isin(types_to_keep)]
        print(f"  Filtered to types {types_to_keep}: {before_count:,} -> {len(result_df):,} rows")

    print(f"  Final type distribution:")
    print(result_df['type'].value_counts().to_string(header=False))

    return result_df


def load_all_corpus_sources(config: dict) -> pd.DataFrame:
    """
    Load and merge all configured corpus sources.

    Args:
        config: Configuration dictionary with 'sources' key

    Returns:
        Combined DataFrame with all sources merged
    """
    print("\n" + "=" * 60)
    print("Loading all corpus sources...")
    print("=" * 60)

    all_dfs = []
    for source_config in config['sources']:
        df = load_corpus_source(source_config)
        all_dfs.append(df)

    # Merge all sources
    print("\n" + "=" * 60)
    print("Merging all sources...")
    print("=" * 60)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    print(f"\nTotal rows from all sources: {len(combined_df):,}")
    print(f"Combined type distribution:")
    print(combined_df['type'].value_counts())

    return combined_df


def save_cache(df: pd.DataFrame, cache_path: Path, config_hash: str = "no_config") -> None:
    """
    Save DataFrame to pickle cache.
    
    Args:
        df: DataFrame to save
        cache_path: Path to cache file
        config_hash: Hash of configuration file for cache validation
    """
    # Create cache directory if it doesn't exist
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with metadata
    cache_data = {
        'data': df,
        'timestamp': datetime.now().isoformat(),
        'version': '1.1',
        'row_count': len(df),
        'config_hash': config_hash
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"\nCache saved to: {cache_path}")
    print(f"Cached {len(df):,} rows")
    print(f"Config hash: {config_hash}")


def load_cache(cache_path: Path, expected_config_hash: str = "no_config") -> Optional[pd.DataFrame]:
    """
    Load DataFrame from pickle cache.
    
    Args:
        cache_path: Path to cache file
        expected_config_hash: Expected config hash for validation
    
    Returns:
        Cached DataFrame or None if cache doesn't exist/is invalid
    """
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Check config hash if present
        cached_hash = cache_data.get('config_hash', 'no_config')
        if cached_hash != expected_config_hash:
            print(f"\nCache config hash mismatch:")
            print(f"  Cached: {cached_hash}")
            print(f"  Current: {expected_config_hash}")
            print(f"Cache invalidated - will rebuild from source...")
            return None
        
        print(f"\n{'='*60}")
        print("Loading from cache...")
        print('='*60)
        print(f"Cache file: {cache_path}")
        print(f"Cached on: {cache_data['timestamp']}")
        print(f"Cache version: {cache_data.get('version', '1.0')}")
        print(f"Rows in cache: {cache_data['row_count']:,}")
        print(f"Config hash: {cached_hash}")
        
        return cache_data['data']
    except Exception as e:
        print(f"\nWarning: Failed to load cache: {e}")
        print("Will rebuild from source data...")
        return None


def clean_data(df: pd.DataFrame, latin_only: bool = False) -> pd.DataFrame:
    """
    Clean the dataset by removing invalid entries.
    
    Args:
        df: Input DataFrame with 'label' and 'type' columns
        latin_only: If True, keep only names with Latin characters (A-Z, a-z, spaces, hyphens, apostrophes)
    
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "=" * 60)
    print("Cleaning data...")
    print("=" * 60)
    
    original_count = len(df)
    print(f"\nOriginal size: {original_count:,} rows")
    
    # Remove rows with null values in label or type columns
    df_clean = df.dropna(subset=["label", "type"]).copy()
    null_removed = original_count - len(df_clean)
    if null_removed > 0:
        print(f"Removed {null_removed:,} rows with null labels or types")
    
    # Remove empty strings (after stripping whitespace)
    df_clean["label"] = df_clean["label"].astype(str).str.strip()
    df_clean = df_clean[df_clean["label"] != ""]
    empty_removed = len(df) - null_removed - len(df_clean)
    if empty_removed > 0:
        print(f"Removed {empty_removed:,} rows with empty/whitespace-only labels")
    
    # Optional: Filter to Latin-only characters
    if latin_only:
        import re
        # Pattern: Latin letters, spaces, hyphens, apostrophes, periods
        latin_pattern = re.compile(r'^[A-Za-z\s\-\'.]+$')
        
        before_latin = len(df_clean)
        df_clean = df_clean[df_clean["label"].str.match(latin_pattern, na=False)]
        non_latin_removed = before_latin - len(df_clean)
        
        if non_latin_removed > 0:
            print(f"Removed {non_latin_removed:,} rows with non-Latin characters")
            print(f"  (Kept only Latin alphabet, spaces, hyphens, apostrophes, periods)")
    
    total_removed = original_count - len(df_clean)
    print(f"\nTotal rows removed: {total_removed:,}")
    print(f"Remaining rows: {len(df_clean):,}")
    
    return df_clean


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
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("corpus/corpus_config.yaml"),
        help="Path to corpus configuration YAML file (default: corpus/corpus_config.yaml)",
    )
    parser.add_argument(
        "--latin-only",
        action="store_true",
        help="Keep only names with Latin characters (removes non-transliterated names)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load configuration
    config = load_corpus_config(args.config_path)
    config_hash = get_config_hash(args.config_path)

    # Try to load from cache if enabled
    df_processed = None
    if args.use_cache and not args.force_rebuild:
        df_processed = load_cache(args.cache_path, config_hash)
    
    # If no cache or cache disabled, load and process from source
    if df_processed is None:
        # Load data based on configuration
        if config:
            # New: Load from multiple sources using configuration
            df = load_all_corpus_sources(config)
        else:
            # Legacy: Load from paranames file only
            print("\nNo configuration file found, using legacy paranames loader...")
            print(f"To use multiple corpus sources, create a config file at: {args.config_path}")
            print(f"See corpus/corpus_config.example.yaml for an example\n")
            df = load_paranames(args.corpus_path)

        # Clean data (remove null, empty, whitespace-only labels)
        df = clean_data(df, latin_only=args.latin_only)
        
        # Filter and deduplicate
        df_processed = filter_and_deduplicate(df)
        
        # Save to cache if enabled
        if args.use_cache:
            save_cache(df_processed, args.cache_path, config_hash)

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
