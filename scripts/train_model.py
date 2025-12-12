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

# Add path to import from name_classifier package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from name_classifier.config import load_model_config, save_model_config


def create_vectorizer_from_config(config: dict):
    """Create a vectorizer instance from configuration dictionary.
    
    Args:
        config: Dictionary containing vectorizer configuration.
        
    Returns:
        Instantiated vectorizer object.
    """
    vectorizer_type = config.get("type", "TfidfVectorizer")
    analyzer = config.get("analyzer", "char")
    ngram_range = tuple(config.get("ngram_range", [2, 4]))
    max_features = config.get("max_features", 10000)
    
    if vectorizer_type == "TfidfVectorizer":
        return TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features
        )
    elif vectorizer_type == "CountVectorizer":
        return CountVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features
        )
    else:
        raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")


def create_model_from_config(config: dict, n_jobs: int = -1):
    """Create a model instance from configuration dictionary.
    
    Args:
        config: Dictionary containing model configuration.
        n_jobs: Number of parallel jobs (for models that support it).
        
    Returns:
        Instantiated model object.
    """
    model_type = config.get("type", "LogisticRegression")
    params = config.get("params", {}).copy()
    
    # Add n_jobs if the model supports it
    if model_type in ["LogisticRegression", "RandomForest"]:
        params["n_jobs"] = n_jobs
    
    if model_type == "LogisticRegression":
        return LogisticRegression(**params)
    elif model_type == "LinearSVC":
        return LinearSVC(**params)
    elif model_type == "RandomForest":
        # Map to actual class name
        return RandomForestClassifier(**params)
    elif model_type == "MultinomialNB":
        return MultinomialNB(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def extract_config_from_result(best_result: dict, vectorizers: dict, classifiers: dict, grid_mode: str) -> dict:
    """Extract configuration dictionary from best result.
    
    Args:
        best_result: Dictionary containing best model results.
        vectorizers: Dictionary of vectorizer instances.
        classifiers: Dictionary of classifier instances.
        grid_mode: Grid search mode used.
        
    Returns:
        Configuration dictionary suitable for saving to YAML.
    """
    vec_name = best_result["vectorizer"]
    clf_name = best_result["classifier"]
    
    # Get vectorizer config
    vectorizer = vectorizers[vec_name]
    vec_config = {
        "type": vectorizer.__class__.__name__,
        "analyzer": vectorizer.analyzer,
        "ngram_range": list(vectorizer.ngram_range),
        "max_features": vectorizer.max_features
    }
    
    # Get model config
    classifier = classifiers[clf_name]
    model_config = {
        "type": clf_name,
        "params": {}
    }
    
    # Extract relevant parameters based on model type
    if clf_name == "LogisticRegression":
        model_config["params"] = {
            "max_iter": classifier.max_iter,
            "random_state": classifier.random_state,
            "class_weight": classifier.class_weight,
            "C": classifier.C
        }
    elif clf_name == "LinearSVC":
        model_config["params"] = {
            "max_iter": classifier.max_iter,
            "random_state": classifier.random_state,
            "class_weight": classifier.class_weight,
            "C": classifier.C
        }
    elif clf_name == "RandomForest":
        model_config["params"] = {
            "n_estimators": classifier.n_estimators,
            "random_state": classifier.random_state,
            "class_weight": classifier.class_weight,
            "max_depth": classifier.max_depth
        }
    elif clf_name == "MultinomialNB":
        model_config["params"] = {
            "alpha": classifier.alpha
        }
    
    # If in parameter grid mode, extract the specific parameters from the result
    if grid_mode == "parameter_grid" and "params" in best_result:
        import ast
        best_params = ast.literal_eval(best_result["params"])
        model_config["params"].update(best_params)
    
    return {
        "vectorizer": vec_config,
        "model": model_config
    }


def should_update_config(new_accuracy: float, old_accuracy: float) -> bool:
    """Determine if config should be updated based on accuracy improvement.
    
    Args:
        new_accuracy: New test accuracy.
        old_accuracy: Old test accuracy from config (can be None).
        
    Returns:
        True if config should be updated, False otherwise.
    """
    if old_accuracy is None:
        return True
    return new_accuracy > old_accuracy



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

    # Filter out rows with NaN values in label or type columns
    train_df_original_len = len(train_df)
    test_df_original_len = len(test_df)
    
    train_df = train_df.dropna(subset=["label", "type"])
    test_df = test_df.dropna(subset=["label", "type"])
    
    # Report filtering results
    train_filtered = train_df_original_len - len(train_df)
    test_filtered = test_df_original_len - len(test_df)
    
    if train_filtered > 0:
        print(f"Warning: Filtered {train_filtered:,} rows with missing data from training set")
    if test_filtered > 0:
        print(f"Warning: Filtered {test_filtered:,} rows with missing data from test set")

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


def get_model_param_grids(n_jobs=-1):
    """Define parameter grids for hyperparameter tuning for each model."""
    return {
        "LogisticRegression": [
            {
                "model": [LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced", n_jobs=n_jobs)],
                "params": {"C": [0.01, 0.1, 1.0, 10.0]},
            }
        ],
        "LinearSVC": [
            {
                "model": [LinearSVC(max_iter=2000, random_state=42, class_weight="balanced")],
                "params": {"C": [0.01, 0.1, 1.0, 10.0]},
            }
        ],
        "RandomForest": [
            {
                "model": [RandomForestClassifier(random_state=42, n_jobs=n_jobs, class_weight="balanced")],
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                },
            }
        ],
        "MultinomialNB": [
            {
                "model": [MultinomialNB()],
                "params": {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]},
            }
        ],
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
    parser.add_argument(
        "--fix-vectorizer",
        type=str,
        default=None,
        help="Fix a specific vectorizer (e.g., 'tfidf_char_2-4') to focus on model comparison",
    )
    parser.add_argument(
        "--exclude-models",
        type=str,
        default=None,
        help="Comma-separated list of models to exclude (e.g., 'RandomForest,MultinomialNB')",
    )
    parser.add_argument(
        "--fix-model",
        type=str,
        default=None,
        help="Fix a specific model (e.g., 'LogisticRegression') to focus on vectorizer comparison or parameter tuning",
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

    # Parse excluded models
    excluded_models = []
    if args.exclude_models:
        excluded_models = [m.strip() for m in args.exclude_models.split(",")]

    # Determine grid search mode
    has_fixed_vectorizer = args.fix_vectorizer is not None
    has_fixed_model = args.fix_model is not None

    if has_fixed_vectorizer and has_fixed_model:
        grid_mode = "parameter_grid"
    elif has_fixed_vectorizer:
        grid_mode = "fixed_vectorizer"
    elif has_fixed_model:
        grid_mode = "fixed_model"
    else:
        grid_mode = "full"

    # Fast train mode: use pre-configured best model from config
    if args.fast_train:
        print("\n" + "=" * 80)
        print("FAST TRAIN MODE ENABLED")
        print("=" * 80)
        
        # Load configuration
        try:
            config = load_model_config()
            fast_train_config = config.get("fast_train", {})
            vec_config = fast_train_config.get("vectorizer", {})
            model_config = fast_train_config.get("model", {})
            
            print(f"Using configuration from YAML:")
            print(f"  Vectorizer: {vec_config.get('type')} {vec_config.get('analyzer')} {vec_config.get('ngram_range')}")
            print(f"  Model: {model_config.get('type')} with params {model_config.get('params', {})}")
            print(f"  Previous accuracy: {fast_train_config.get('accuracy', {}).get('test_accuracy', 'N/A')}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load config ({e}), using defaults")
            # Fallback to defaults
            vec_config = {
                "type": "TfidfVectorizer",
                "analyzer": "char",
                "ngram_range": [2, 4],
                "max_features": 10000
            }
            model_config = {
                "type": "LogisticRegression",
                "params": {
                    "max_iter": 1000,
                    "random_state": 42,
                    "class_weight": "balanced",
                    "C": 1.0
                }
            }
        
        print(f"Multiprocessing: {args.n_jobs} jobs")
        print("=" * 80)
        
        # Create vectorizer and classifier from config
        vectorizer = create_vectorizer_from_config(vec_config)
        classifier = create_model_from_config(model_config, n_jobs=args.n_jobs)
        
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
        
        # Update config with new accuracy
        try:
            config = load_model_config()
        except:
            config = {"fast_train": {}}
        
        if "fast_train" not in config:
            config["fast_train"] = {}
        
        config["fast_train"]["vectorizer"] = vec_config
        config["fast_train"]["model"] = model_config
        config["fast_train"]["accuracy"] = {
            "test_accuracy": float(metrics['test_accuracy']),
            "test_f1": float(metrics['test_f1']),
            "cv_mean": float(metrics['cv_mean']),
            "cv_std": float(metrics['cv_std'])
        }
        config["fast_train"]["last_updated"] = datetime.now().isoformat()
        config["fast_train"]["training_date"] = datetime.now().isoformat()
        
        save_model_config(config)
        print(f"\nConfig updated with accuracy: {metrics['test_accuracy']:.4f}")
        
        # Prepare results for saving (for metadata.json)
        best_result = {
            "vectorizer": vec_config.get("type", "TfidfVectorizer"),
            "classifier": model_config.get("type", "LogisticRegression"),
            **metrics,
        }
        best_vectorizer = vectorizer
        best_classifier = classifier
        
    else:
        # Grid search modes
        print("\n" + "=" * 80)
        print(f"GRID SEARCH MODE: {grid_mode.upper().replace('_', ' ')}")
        print("=" * 80)
        if has_fixed_vectorizer:
            print(f"Fixed Vectorizer: {args.fix_vectorizer}")
        if has_fixed_model:
            print(f"Fixed Model: {args.fix_model}")
        if excluded_models:
            print(f"Excluded Models: {', '.join(excluded_models)}")
        print(f"Multiprocessing: {args.n_jobs} jobs")
        print("=" * 80)

        # Get configurations
        all_vectorizers = get_vectorizers()
        all_classifiers = get_classifiers(n_jobs=args.n_jobs)

        # Validate fixed vectorizer if specified
        if has_fixed_vectorizer and args.fix_vectorizer not in all_vectorizers:
            raise ValueError(
                f"Invalid vectorizer '{args.fix_vectorizer}'. "
                f"Available: {', '.join(all_vectorizers.keys())}"
            )

        # Validate fixed model if specified
        if has_fixed_model and args.fix_model not in all_classifiers:
            raise ValueError(
                f"Invalid model '{args.fix_model}'. "
                f"Available: {', '.join(all_classifiers.keys())}"
            )

        # Validate excluded models
        for model in excluded_models:
            if model not in all_classifiers:
                raise ValueError(
                    f"Invalid excluded model '{model}'. "
                    f"Available: {', '.join(all_classifiers.keys())}"
                )

        # Filter vectorizers and classifiers based on mode
        if has_fixed_vectorizer:
            vectorizers = {args.fix_vectorizer: all_vectorizers[args.fix_vectorizer]}
        else:
            vectorizers = all_vectorizers

        if has_fixed_model:
            classifiers = {args.fix_model: all_classifiers[args.fix_model]}
        else:
            classifiers = {k: v for k, v in all_classifiers.items() if k not in excluded_models}

        # Results storage
        results = []

        # Parameter grid mode: Test hyperparameters for a fixed model + vectorizer
        if grid_mode == "parameter_grid":
            param_grids = get_model_param_grids(n_jobs=args.n_jobs)
            param_grid = param_grids[args.fix_model]

            # Fit vectorizer
            vectorizer = vectorizers[args.fix_vectorizer]
            print(f"\nFitting vectorizer: {args.fix_vectorizer}...")
            X_train = vectorizer.fit_transform(X_train_text)
            X_test = vectorizer.transform(X_test_text)

            # Test each parameter configuration
            print(f"\nTesting {args.fix_model} with parameter grid...")
            for grid_config in param_grid:
                base_model = grid_config["model"][0]
                params_dict = grid_config["params"]

                # Generate all parameter combinations
                import itertools
                param_names = list(params_dict.keys())
                param_values = list(params_dict.values())
                param_combinations = list(itertools.product(*param_values))

                total_combinations = len(param_combinations)
                for idx, param_combo in enumerate(param_combinations, 1):
                    # Create model with current parameters
                    params = dict(zip(param_names, param_combo))
                    from sklearn.base import clone
                    model = clone(base_model)
                    model.set_params(**params)

                    print(f"\n[{idx}/{total_combinations}] Testing parameters: {params}")

                    # Evaluate
                    metrics = evaluate_model(
                        model, X_train, y_train, X_test, y_test, cv=args.cv_folds, n_jobs=args.n_jobs
                    )

                    result = {
                        "vectorizer": args.fix_vectorizer,
                        "classifier": args.fix_model,
                        "params": str(params),
                        **metrics,
                    }
                    results.append(result)

                    print(f"  CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
                    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
                    print(f"  Test F1 (weighted): {metrics['test_f1']:.4f}")
                    print(f"  Test F1 ORG: {metrics['test_f1_org']:.4f}")
                    print(f"  Test F1 PER: {metrics['test_f1_per']:.4f}")
                    print(f"  Train Time: {metrics['train_time']:.2f}s")

        else:
            # Model vs Vectorizer grid mode
            print("\nTraining and evaluating models...")
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
        if grid_mode == "parameter_grid":
            print(
                results_df[
                    ["vectorizer", "classifier", "params", "cv_mean", "test_accuracy", "test_f1", "train_time"]
                ].head(10).to_string(index=False)
            )
        else:
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
        if grid_mode == "parameter_grid":
            print(f"Best Parameters: {best_result['params']}")
        print(f"Test Accuracy: {best_result['test_accuracy']:.4f}")
        print(f"Test F1 Score: {best_result['test_f1']:.4f}")

        # Train final model with best configuration
        print("\n" + "=" * 80)
        print("Training final model with best configuration...")
        print("=" * 80)

        best_vectorizer = vectorizers[best_result["vectorizer"]]
        
        # Recreate best classifier with parameters if in parameter grid mode
        if grid_mode == "parameter_grid":
            import ast
            best_params = ast.literal_eval(best_result["params"])
            param_grids = get_model_param_grids(n_jobs=args.n_jobs)
            base_model = param_grids[best_result["classifier"]][0]["model"][0]
            from sklearn.base import clone
            best_classifier = clone(base_model)
            best_classifier.set_params(**best_params)
        else:
            best_classifier = classifiers[best_result["classifier"]]
        
        # Check if we should update the config with better results
        try:
            config = load_model_config()
            current_accuracy = config.get("fast_train", {}).get("accuracy", {}).get("test_accuracy")
            new_accuracy = float(best_result["test_accuracy"])
            
            if should_update_config(new_accuracy, current_accuracy):
                print("\n" + "=" * 80)
                print("UPDATING CONFIGURATION")
                print("=" * 80)
                print(f"New accuracy ({new_accuracy:.4f}) is better than config ({current_accuracy})")
                print("Updating model_config.yaml with new best configuration...")
                
                # Extract config from best result
                new_config = extract_config_from_result(best_result, vectorizers, classifiers, grid_mode)
                
                # Update the config
                if "fast_train" not in config:
                    config["fast_train"] = {}
                
                config["fast_train"]["vectorizer"] = new_config["vectorizer"]
                config["fast_train"]["model"] = new_config["model"]
                config["fast_train"]["accuracy"] = {
                    "test_accuracy": float(best_result["test_accuracy"]),
                    "test_f1": float(best_result["test_f1"]),
                    "cv_mean": float(best_result["cv_mean"]),
                    "cv_std": float(best_result["cv_std"])
                }
                config["fast_train"]["last_updated"] = datetime.now().isoformat()
                config["fast_train"]["training_date"] = datetime.now().isoformat()
                
                save_model_config(config)
                print("Configuration updated successfully!")
                print("=" * 80)
            else:
                print(f"\nCurrent config accuracy ({current_accuracy:.4f}) is better or equal. Not updating config.")
        except Exception as e:
            print(f"\nWarning: Could not update config: {e}")


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
        "vectorizer": best_result["vectorizer"] if isinstance(best_result["vectorizer"], str) else str(best_result["vectorizer"]),
        "classifier": best_result["classifier"] if isinstance(best_result["classifier"], str) else str(best_result["classifier"]),
        "test_accuracy": float(best_result["test_accuracy"]),
        "test_f1": float(best_result["test_f1"]),
        "cv_accuracy_mean": float(best_result["cv_mean"]),
        "cv_accuracy_std": float(best_result["cv_std"]),
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "fast_train_mode": args.fast_train,
        "n_jobs": args.n_jobs,
        "grid_mode": grid_mode if not args.fast_train else "fast_train",
        "fixed_vectorizer": args.fix_vectorizer,
        "fixed_model": args.fix_model,
        "excluded_models": excluded_models if excluded_models else None,
    }
    
    # Add best parameters if in parameter grid mode
    if not args.fast_train and grid_mode == "parameter_grid":
        metadata["best_params"] = best_result["params"] if isinstance(best_result["params"], str) else str(best_result["params"])

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
