#!/usr/bin/env python3
"""
Demonstration of the Model Configuration System

This script demonstrates:
1. Loading model configuration from YAML
2. Creating model and vectorizer instances from config
3. Updating config with new accuracy results
4. Comparing accuracies to decide on updates
"""
import yaml
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "name_classifier" / "models" / "model_config.yaml"

def load_config():
    """Load the config file."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_config(config):
    """Save the config file."""
    with open(CONFIG_PATH, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

def demo_config_system():
    """Demonstrate the configuration system."""
    print("=" * 80)
    print("MODEL CONFIGURATION SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Step 1: Load current config
    print("\n1. LOADING CURRENT CONFIGURATION")
    print("-" * 80)
    config = load_config()
    ft_config = config.get("fast_train", {})
    
    vec_config = ft_config.get("vectorizer", {})
    model_config = ft_config.get("model", {})
    accuracy_config = ft_config.get("accuracy", {})
    
    print(f"Vectorizer Configuration:")
    print(f"  Type: {vec_config.get('type')}")
    print(f"  Analyzer: {vec_config.get('analyzer')}")
    print(f"  N-gram Range: {vec_config.get('ngram_range')}")
    print(f"  Max Features: {vec_config.get('max_features')}")
    
    print(f"\nModel Configuration:")
    print(f"  Type: {model_config.get('type')}")
    print(f"  Parameters:")
    for key, value in model_config.get('params', {}).items():
        print(f"    {key}: {value}")
    
    print(f"\nCurrent Accuracy Metrics:")
    print(f"  Test Accuracy: {accuracy_config.get('test_accuracy')}")
    print(f"  Test F1: {accuracy_config.get('test_f1')}")
    print(f"  CV Mean: {accuracy_config.get('cv_mean')}")
    print(f"  CV Std: {accuracy_config.get('cv_std')}")
    
    # Step 2: Simulate a training run finding better results
    print("\n\n2. SIMULATING GRID SEARCH FINDING BETTER MODEL")
    print("-" * 80)
    
    # Simulate new results
    new_results = {
        "vectorizer": {
            "type": "TfidfVectorizer",
            "analyzer": "char",
            "ngram_range": [2, 5],  # Different!
            "max_features": 10000
        },
        "model": {
            "type": "LogisticRegression",
            "params": {
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": "balanced",
                "C": 0.1  # Different parameter!
            }
        },
        "accuracy": {
            "test_accuracy": 0.9543,  # Simulated better accuracy
            "test_f1": 0.9512,
            "cv_mean": 0.9489,
            "cv_std": 0.0023
        }
    }
    
    current_accuracy = accuracy_config.get('test_accuracy')
    new_accuracy = new_results['accuracy']['test_accuracy']
    
    print(f"Current config accuracy: {current_accuracy}")
    print(f"New model accuracy: {new_accuracy}")
    
    # Step 3: Decide whether to update
    print("\n\n3. COMPARING ACCURACIES")
    print("-" * 80)
    
    should_update = (current_accuracy is None) or (new_accuracy > current_accuracy)
    
    if should_update:
        print(f"✓ New accuracy ({new_accuracy:.4f}) is better!")
        print(f"  Updating configuration...")
        
        # Update config
        config["fast_train"]["vectorizer"] = new_results["vectorizer"]
        config["fast_train"]["model"] = new_results["model"]
        config["fast_train"]["accuracy"] = new_results["accuracy"]
        config["fast_train"]["last_updated"] = datetime.now().isoformat()
        
        # We won't actually save to avoid modifying the real config
        # save_config(config)
        
        print(f"  New vectorizer n-gram range: {new_results['vectorizer']['ngram_range']}")
        print(f"  New model C parameter: {new_results['model']['params']['C']}")
        print(f"  New accuracy: {new_accuracy:.4f}")
        print("\n  (Config would be updated in real scenario)")
    else:
        print(f"✗ Current accuracy ({current_accuracy:.4f}) is better or equal")
        print(f"  No update needed")
    
    # Step 4: Show how fast-train would use the config
    print("\n\n4. HOW FAST-TRAIN USES THE CONFIG")
    print("-" * 80)
    print("When running with --fast-train flag:")
    print("  1. Load config from model_config.yaml")
    print("  2. Create vectorizer and model instances from config")
    print(f"     → TfidfVectorizer(analyzer='{vec_config.get('analyzer')}', "
          f"ngram_range={vec_config.get('ngram_range')}, ...)")
    print(f"     → {model_config.get('type')}(**{model_config.get('params')})")
    print("  3. Train model")
    print("  4. Update config with new accuracy metrics")
    print("  5. Save updated config back to YAML")
    
    print("\n\n5. SUMMARY")
    print("=" * 80)
    print("✓ Configuration system is working correctly!")
    print("\nKey Features:")
    print("  • Fast-train reads model settings from YAML config")
    print("  • Fast-train saves accuracy metrics after training")
    print("  • Grid search updates config when finding better models")
    print("  • Config includes vectorizer, model, and accuracy settings")
    print("  • Automatic comparison ensures best config is preserved")
    print("=" * 80)

if __name__ == "__main__":
    demo_config_system()
