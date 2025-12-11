#!/usr/bin/env python
"""
Model training script for the name classifier.

This script:
1. Loads prepared training data
2. Experiments with multiple vectorizer configurations
3. Trains multiple classifiers
4. Evaluates using cross-validation
5. Compares models and selects the best performer
6. Trains final model on full training set
7. Saves best model, vectorizer, and metadata
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report


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
            f"Test data not found at {test_path}. " f"Please run scripts/prepare_data.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded training data: {len(train_df):,} samples")
    print(f"Loaded test data: {len(test_df):,} samples")

    return train_df, test_df


def get_vectorizers():
    """Define vectorizer configurations to test."""
    return {
        "tfidf_char_2-4": TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), max_features=10000
        ),
        "tfidf_char_2-5": TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), max_features=10000
        ),
        "tfidf_char_3-5": TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), max_features=10000
        ),
        "tfidf_word_1-2": TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=10000
        ),
        "tfidf_word_1-3": TfidfVectorizer(
            analyzer="word", ngram_range=(1, 3), max_features=10000
        ),
        "count_char_2-4": CountVectorizer(
            analyzer="char", ngram_range=(2, 4), max_features=10000
        ),
    }


def get_classifiers(n_jobs=-1):
    """Define classifiers to test with balanced class weights for imbalanced data."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced", n_jobs=n_jobs
        ),
        "LinearSVC": LinearSVC(max_iter=2000, random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=n_jobs, class_weight="balanced"
        ),
        "MultinomialNB": MultinomialNB(),  # NB doesn't support class_weight
    }


def evaluate_model(classifier, X_train, y_train, X_test, y_test, cv=3, n_jobs=-1):
    """Evaluate a model using cross-validation and test set."""
    # Cross-validation on training set
    cv_scores = cross_val_score(
        classifier, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=n_jobs
    )
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Train on full training set and evaluate on test set
    start_time = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    # Per-class F1 scores to monitor class imbalance handling
    test_f1_per_class = f1_score(y_test, y_pred, average=None, labels=["ORG", "PER"])

    return {
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1_weighted,
        "test_f1_org": test_f1_per_class[0],  # ORG F1
        "test_f1_per": test_f1_per_class[1],  # PER F1
        "train_time": train_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Train name classifier models")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train.csv and test.csv (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("name_classifier/models"),
        help="Output directory for trained models (default: name_classifier/models/)",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=3, help="Number of cross-validation folds (default: 3)"
    )
    parser.add_argument(
        "--fast-train",
        action="store_true",
        help="Use pre-configured fast model (char-ngram TF-IDF + LogisticRegression) instead of grid search",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for multiprocessing (default: -1 for all cores)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 80)
    print("Loading data...")
    print("=" * 80)
    train_df, test_df = load_data(args.data_dir)

    X_train_text = train_df["label"].values
    y_train = train_df["type"].values
    X_test_text = test_df["label"].values
    y_test = test_df["type"].values

    # Fast train mode: use pre-configured best model
    if args.fast_train:
        print("\n" + "=" * 80)
        print("FAST TRAIN MODE ENABLED")
        print("=" * 80)
        print("Using pre-configured model: TF-IDF char(2-4) + LogisticRegression")
        print(f"Multiprocessing: {args.n_jobs} jobs")
        print("=" * 80)
        
        # Use best performing configuration from benchmarks
        vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), max_features=10000
        )
        classifier = LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced", n_jobs=args.n_jobs
        )
        
        # Fit vectorizer
        print("\nFitting vectorizer...")
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
        
        # Evaluate
        print("\nTraining and evaluating model...")
        metrics = evaluate_model(
            classifier, X_train, y_train, X_test, y_test, cv=args.cv_folds, n_jobs=args.n_jobs
        )
        
        print("\n" + "=" * 80)
        print("FAST TRAIN RESULTS")
        print("=" * 80)
        print(f"CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Test F1 (weighted): {metrics['test_f1']:.4f}")
        print(f"Test F1 ORG: {metrics['test_f1_org']:.4f}")
        print(f"Test F1 PER: {metrics['test_f1_per']:.4f}")
        print(f"Train Time: {metrics['train_time']:.2f}s")
        
        # Prepare results for saving
        best_result = {
            "vectorizer": "tfidf_char_2-4",
            "classifier": "LogisticRegression",
            **metrics,
        }
        best_vectorizer = vectorizer
        best_classifier = classifier
        
    else:
        # Full grid search mode
        # Get configurations
        vectorizers = get_vectorizers()
        classifiers = get_classifiers(n_jobs=args.n_jobs)

        # Results storage
        results = []

        # Train and evaluate all combinations
        print("\n" + "=" * 80)
        print("Training and evaluating models...")
        print(f"Multiprocessing: {args.n_jobs} jobs")
        print("=" * 80)

        total_combinations = len(vectorizers) * len(classifiers)
        current = 0

        for vec_name, vectorizer in vectorizers.items():
            print(f"\n--- Vectorizer: {vec_name} ---")

            # Fit vectorizer on training data
            print(f"Fitting vectorizer...")
            X_train = vectorizer.fit_transform(X_train_text)
            X_test = vectorizer.transform(X_test_text)

            for clf_name, classifier in classifiers.items():
                current += 1
                print(
                    f"\n[{current}/{total_combinations}] Testing: {vec_name} + {clf_name}"
                )

                # Evaluate
                metrics = evaluate_model(
                    classifier, X_train, y_train, X_test, y_test, cv=args.cv_folds, n_jobs=args.n_jobs
                )

                result = {
                    "vectorizer": vec_name,
                    "classifier": clf_name,
                    **metrics,
                }
                results.append(result)

                print(
                    f"  CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})"
                )
                print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
                print(f"  Test F1 (weighted): {metrics['test_f1']:.4f}")
                print(f"  Test F1 ORG: {metrics['test_f1_org']:.4f}")
                print(f"  Test F1 PER: {metrics['test_f1_per']:.4f}")
                print(f"  Train Time: {metrics['train_time']:.2f}s")

        # Display results summary
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("test_accuracy", ascending=False)

        print("\nTop 10 Models (sorted by test accuracy):")
        print(
            results_df[
                ["vectorizer", "classifier", "cv_mean", "test_accuracy", "test_f1", "train_time"]
            ].head(10).to_string(index=False)
        )

        # Select best model
        best_result = results_df.iloc[0]
        print("\n" + "=" * 80)
        print("BEST MODEL")
        print("=" * 80)
        print(f"Vectorizer: {best_result['vectorizer']}")
        print(f"Classifier: {best_result['classifier']}")
        print(f"Test Accuracy: {best_result['test_accuracy']:.4f}")
        print(f"Test F1 Score: {best_result['test_f1']:.4f}")

        # Train final model with best configuration
        print("\n" + "=" * 80)
        print("Training final model with best configuration...")
        print("=" * 80)

        best_vectorizer = vectorizers[best_result["vectorizer"]]
        best_classifier = classifiers[best_result["classifier"]]

    # Fit on training data
    X_train_final = best_vectorizer.fit_transform(X_train_text)
    best_classifier.fit(X_train_final, y_train)

    # Test set evaluation
    X_test_final = best_vectorizer.transform(X_test_text)
    y_pred_final = best_classifier.predict(X_test_final)

    print("\nFinal Model Performance on Test Set:")
    print(classification_report(y_test, y_pred_final))

    # Save model artifacts
    print("\n" + "=" * 80)
    print("Saving model artifacts...")
    print("=" * 80)

    model_path = args.output_dir / "model.pkl"
    vectorizer_path = args.output_dir / "vectorizer.pkl"
    metadata_path = args.output_dir / "metadata.json"

    joblib.dump(best_classifier, model_path)
    print(f"Model saved to: {model_path}")

    joblib.dump(best_vectorizer, vectorizer_path)
    print(f"Vectorizer saved to: {vectorizer_path}")

    # Save metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "vectorizer": best_result["vectorizer"],
        "classifier": best_result["classifier"],
        "test_accuracy": float(best_result["test_accuracy"]),
        "test_f1": float(best_result["test_f1"]),
        "cv_accuracy_mean": float(best_result["cv_mean"]),
        "cv_accuracy_std": float(best_result["cv_std"]),
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "fast_train_mode": args.fast_train,
        "n_jobs": args.n_jobs,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run tests: pytest")
    print("  2. Try the classifier:")
    print('     python -c "from name_classifier import classify; print(classify(\'Bob Smith\'))"')


if __name__ == "__main__":
    main()
