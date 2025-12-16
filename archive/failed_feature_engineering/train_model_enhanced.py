#!/usr/bin/env python
"""
Enhanced model training script with ISO 20275 features and engineered features.

This script trains a name classifier using:
1. Character n-grams via HashingVectorizer (sparse, ~1M features)
2. Engineered features via NameFeatureExtractor (dense, 32 features)
3. Tier-conditional feature dropout during training
4. Focused models: SGDClassifier and LogisticRegression only
5. Comprehensive evaluation slicing
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_curve

# Add path to import from name_classifier package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from name_classifier.transformers import NameFeatureExtractor, TextExtractor
from name_classifier.feature_dropout import apply_feature_dropout, extract_tier_metadata
from name_classifier.iso20275_matcher import ISO20275Matcher
from name_classifier.config import load_model_config, save_model_config


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data."""
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            f"Please run scripts/prepare_data.py first."
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}. "
            f"Please run scripts/prepare_data.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Filter out rows with NaN values
    train_df = train_df.dropna(subset=["label", "type"])
    test_df = test_df.dropna(subset=["label", "type"])

    print(f"Loaded training data: {len(train_df):,} samples")
    print(f"Loaded test data: {len(test_df):,} samples")

    return train_df, test_df


def create_feature_pipeline(hash_features: int = 2**20, language_col: str = None):
    """Create hybrid feature extraction pipeline.
    
    Args:
        hash_features: Number of hash features for char n-grams (default: 2^20 = 1,048,576)
        language_col: Optional language column name for hints
        
    Returns:
        FeatureUnion pipeline combining char n-grams + engineered features
    """
    # Character n-gram vectorizer (sparse)
    char_ngrams = Pipeline([
        ('text_extract', TextExtractor(column='label')),
        ('hashing', HashingVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            n_features=hash_features,
            norm='l2'
        ))
    ])
    
    # Engineered features (dense, 32 features)
    engineered_features = NameFeatureExtractor(
        language_column=language_col
    )
    
    #  Combine both
    feature_union = FeatureUnion([
        ('char_ngrams', char_ngrams),
        ('engineered', engineered_features)
    ])
    
    return feature_union


def get_classifiers(n_jobs=-1):
    """Get focused set of classifiers for training.
    
    Returns only SGDClassifier and LogisticRegression as per requirements.
    Optimized for large datasets (millions of samples).
    """
    return {
        "SGDClassifier": SGDClassifier(
            loss='log_loss',
            max_iter=2000,  # Increased for better convergence
            random_state=42,
            class_weight='balanced',
            alpha=0.0001,  # L2 regularization
            learning_rate='optimal',  # Adaptive learning rate
            early_stopping=True,  # Stop early if not improving
            validation_fraction=0.1,
            n_iter_no_change=5,
            n_jobs=n_jobs,
            verbose=0
        ),
        "LogisticRegression": LogisticRegression(
            solver='saga',
            max_iter=100,  # Reduced from 1000 - SAGA is slow on huge datasets
            random_state=42,
            class_weight='balanced',
            tol=0.01,  # Less strict convergence for speed
            verbose=1  # Show progress
        )
    }


def evaluate_model(classifier, X_train, y_train, X_test, y_test, cv=3, n_jobs=-1):
    """Evaluate a model using cross-validation and test set."""
    # Cross-validation on training set
    cv_scores = cross_val_score(
        classifier, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=n_jobs
    )
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Train on full training set
    start_time = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluate on test set
    y_pred = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    # Per-class F1 scores
    test_f1_per_class = f1_score(y_test, y_pred, average=None, labels=["ORG", "PER"])

    return {
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1_weighted,
        "test_f1_org": test_f1_per_class[0],
        "test_f1_per": test_f1_per_class[1],
        "train_time": train_time,
    }


def evaluate_slices(y_true, y_pred, y_prob, names, iso_matcher):
    """Evaluate model performance on different data slices.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (n_samples, 2) for ORG class
        names: List of name strings
        iso_matcher: ISO20275Matcher instance
        
    Returns:
        Dictionary of slice metrics
    """
    from sklearn.metrics import precision_score, recall_score
    
    slices = {}
    
    # Extract suffix matches for all names
    suffix_matches = [iso_matcher.match_legal_form(name) for name in names]
    
    # 1. By token length
    token_lengths = [len(name.split()) for name in names]
    for length_bin in [1, 2, '3+']:
        if length_bin == '3+':
            mask = np.array([t >= 3 for t in token_lengths])
        else:
            mask = np.array([t == length_bin for t in token_lengths])
        
        if mask.sum() > 0:
            slices[f'token_len_{length_bin}'] = {
                'count': int(mask.sum()),
                'accuracy': float(accuracy_score(y_true[mask], y_pred[mask])),
                'f1': float(f1_score(y_true[mask], y_pred[mask], average='weighted'))
            }
    
    # 2. By suffix presence
    has_suffix_mask = np.array([m is not None for m in suffix_matches])
    slices['has_suffix'] = {
        'count': int(has_suffix_mask.sum()),
        'accuracy': float(accuracy_score(y_true[has_suffix_mask], y_pred[has_suffix_mask])) if has_suffix_mask.sum() > 0 else 0.0,
        'f1': float(f1_score(y_true[has_suffix_mask], y_pred[has_suffix_mask], average='weighted')) if has_suffix_mask.sum() > 0 else 0.0
    }
    
    slices['no_suffix'] = {
        'count': int((~has_suffix_mask).sum()),
        'accuracy': float(accuracy_score(y_true[~has_suffix_mask], y_pred[~has_suffix_mask])) if (~has_suffix_mask).sum() > 0 else 0.0,
        'f1': float(f1_score(y_true[~has_suffix_mask], y_pred[~has_suffix_mask], average='weighted')) if (~has_suffix_mask).sum() > 0 else 0.0
    }
    
    # 3. By tier (ambiguous short forms - Tier C only)
    tier_c_mask = np.array([m is not None and m.metadata.tier == 'C' for m in suffix_matches])
    if tier_c_mask.sum() > 0:
        slices['tier_c_ambiguous'] = {
            'count': int(tier_c_mask.sum()),
            'accuracy': float(accuracy_score(y_true[tier_c_mask], y_pred[tier_c_mask])),
            'f1': float(f1_score(y_true[tier_c_mask], y_pred[tier_c_mask], average='weighted'))
        }
    
    # 4. Precision/Recall at thresholds
    # Get probabilities for ORG class
    if y_prob.ndim == 2:
        org_probs = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob[:, 0]
    else:
        org_probs = y_prob
    
    y_true_binary = (y_true == 'ORG').astype(int)
    
    slices['threshold_metrics'] = {}
    for threshold in [0.5, 0.7, 0.9]:
        y_pred_thresh = (org_probs >= threshold).astype(int)
        if y_pred_thresh.sum() > 0:
            slices['threshold_metrics'][f'threshold_{threshold}'] = {
                'precision': float(precision_score(y_true_binary, y_pred_thresh, zero_division=0)),
                'recall': float(recall_score(y_true_binary, y_pred_thresh, zero_division=0)),
                'count_predicted_org': int(y_pred_thresh.sum())
            }
    
    return slices


def main():
    parser = argparse.ArgumentParser(description="Train enhanced name classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train.csv and test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("name_classifier/models"),
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--cv-folds", 
        type=int, 
        default=3, 
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--hash-features",
        type=int,
        default=2**18,  # Reduced from 2^20 - 262k is plenty for most cases
        help="Number of hash features for char n-grams (default: 2^18=262k, try 2^16=65k for speed)",
    )
    parser.add_argument(
        "--dropout-tier-ab",
        type=float,
        default=0.3,
        help="Dropout rate for Tier A/B legal forms (default: 0.3)",
    )
    parser.add_argument(
        "--dropout-tier-c",
        type=float,
        default=0.5,
        help="Dropout rate for Tier C ambiguous forms (default: 0.5)",
    )
    parser.add_argument(
        "--no-dropout",
        action="store_true",
        help="Disable feature dropout during training",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ENHANCED NAME CLASSIFIER TRAINING")
    print("=" * 80)
    print(f"Hash features: {args.hash_features:,}")
    print(f"Feature dropout: {'DISABLED' if args.no_dropout else f'Tier A/B={args.dropout_tier_ab}, Tier C={args.dropout_tier_c}'}")
    print(f"Models: SGDClassifier, LogisticRegression")
    print(f"n_jobs: {args.n_jobs}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    train_df, test_df = load_data(args.data_dir)

    # Check for language column
    language_col = 'language' if 'language' in train_df.columns else None
    if language_col:
        print(f"Using language hints from '{language_col}' column")

    # Create feature pipeline
    print("\nCreating feature extraction pipeline...")
    feature_pipeline = create_feature_pipeline(
        hash_features=args.hash_features,
        language_col=language_col
    )

    # Fit pipeline and transform data
    print("Extracting features from training data...")
    X_train = feature_pipeline.fit_transform(train_df)
    print(f"Training features shape: {X_train.shape}")
    
    print("Extracting features from test data...")
    X_test = feature_pipeline.transform(test_df)
    print(f"Test features shape: {X_test.shape}")

    y_train = train_df["type"].values
    y_test = test_df["type"].values

    # Apply feature dropout if enabled
    if not args.no_dropout:
        print("\nApplying tier-conditional feature dropout to training data...")
        iso_matcher = ISO20275Matcher()
        tier_metadata = extract_tier_metadata(train_df["label"].values, iso_matcher)
        
        # Note: This applies to the engineered features (last 32 columns)
        # We need to extract just those columns, apply dropout, and recombine
        X_train_sparse = X_train[:, :-32]  # Char n-grams
        X_train_dense = X_train[:, -32:].toarray() if hasattr(X_train[:, -32:], 'toarray') else X_train[:, -32:]  # Engineered features
        
        X_train_dense_dropped = apply_feature_dropout(
            X_train_dense,
            tier_metadata,
            tier_ab_rate=args.dropout_tier_ab,
            tier_c_rate=args.dropout_tier_c,
            random_state=42
        )
        
        # Recombine
        from scipy.sparse import hstack, csr_matrix
        X_train = hstack([X_train_sparse, csr_matrix(X_train_dense_dropped)])
        print(f"Dropout applied. Final training shape: {X_train.shape}")

    # Train models
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)

    classifiers = get_classifiers(n_jobs=args.n_jobs)
    results = []

    for clf_name, classifier in classifiers.items():
        print(f"\nTraining {clf_name}...")
        
        metrics = evaluate_model(
            classifier,
            X_train,
            y_train,
            X_test,
            y_test,
            cv=args.cv_folds,
            n_jobs=args.n_jobs
        )

        result = {"classifier": clf_name, **metrics}
        results.append(result)

        print(f"  CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Test F1: {metrics['test_f1']:.4f}")
        print(f"  Train Time: {metrics['train_time']:.2f}s")

    # Select best model
    results_df = pd.DataFrame(results)
    best_idx = results_df['test_accuracy'].idxmax()
    best_result = results_df.iloc[best_idx]

    print("\n" + "=" * 80)
    print("BEST MODEL")
    print("=" * 80)
    print(f"Classifier: {best_result['classifier']}")
    print(f"Test Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"Test F1: {best_result['test_f1']:.4f}")

    # Re-train best model for final version
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)
    
    best_classifier_name = best_result['classifier']
    final_classifier = classifiers[best_classifier_name]
    final_classifier.fit(X_train, y_train)
    
    # Final predictions
    y_pred_final = final_classifier.predict(X_test)
    y_prob_final = final_classifier.predict_proba(X_test) if hasattr(final_classifier, 'predict_proba') else None

    print("\nFinal Model Performance:")
    print(classification_report(y_test, y_pred_final))

    # Evaluation slicing
    print("\n" + "=" * 80)
    print("EVALUATION SLICING")
    print("=" * 80)
    
    iso_matcher = ISO20275Matcher()
    slices = evaluate_slices(
        y_test,
        y_pred_final,
        y_prob_final,
        test_df["label"].values,
        iso_matcher
    )

    print("\nPerformance by Token Length:")
    for key in ['token_len_1', 'token_len_2', 'token_len_3+']:
        if key in slices:
            print(f"  {key}: Acc={slices[key]['accuracy']:.4f}, Count={slices[key]['count']}")

    print("\nPerformance by Suffix Presence:")
    print(f"  Has suffix: Acc={slices['has_suffix']['accuracy']:.4f}, Count={slices['has_suffix']['count']}")
    print(f"  No suffix: Acc={slices['no_suffix']['accuracy']:.4f}, Count={slices['no_suffix']['count']}")

    if 'tier_c_ambiguous' in slices:
        print("\nPerformance on Ambiguous Forms (Tier C):")
        print(f"  Accuracy: {slices['tier_c_ambiguous']['accuracy']:.4f}, Count={slices['tier_c_ambiguous']['count']}")

    # Save artifacts
    print("\n" + "=" * 80)
    print("SAVING ARTIFACTS")
    print("=" * 80)

    model_path = args.output_dir / "model.pkl"
    pipeline_path = args.output_dir / "feature_pipeline.pkl"
    metadata_path = args.output_dir / "metadata.json"

    joblib.dump(final_classifier, model_path)
    print(f"Model saved to: {model_path}")

    joblib.dump(feature_pipeline, pipeline_path)
    print(f"Feature pipeline saved to: {pipeline_path}")

    # Save comprehensive metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "classifier": best_classifier_name,
        "test_accuracy": float(best_result["test_accuracy"]),
        "test_f1": float(best_result["test_f1"]),
        "cv_accuracy_mean": float(best_result["cv_mean"]),
        "cv_accuracy_std": float(best_result["cv_std"]),
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "hash_features": args.hash_features,
        "feature_dropout_enabled": not args.no_dropout,
        "dropout_tier_ab": args.dropout_tier_ab if not args.no_dropout else None,
        "dropout_tier_c": args.dropout_tier_c if not args.no_dropout else None,
        "evaluation_slices": slices,
        "feature_count": {
            "total": X_train.shape[1],
            "char_ngrams": X_train.shape[1] - 32,
            "engineered": 32
        }
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
