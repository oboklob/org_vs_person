#!/usr/bin/env python3
"""
Simple test script to verify config loading and saving functionality.
"""
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from name_classifier.config import load_model_config, save_model_config, MODEL_CONFIG_PATH

def test_config_operations():
    """Test config loading and saving."""
    print("=" * 60)
    print("Testing Model Configuration System")
    print("=" * 60)
    
    # Test 1: Load config
    print("\n1. Loading config from:", MODEL_CONFIG_PATH)
    try:
        config = load_model_config()
        print("✓ Config loaded successfully")
        print("\nConfig structure:")
        import yaml
        print(yaml.safe_dump(config, default_flow_style=False))
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False
    
    # Test 2: Verify structure
    print("\n2. Verifying config structure...")
    if "fast_train" not in config:
        print("✗ Missing 'fast_train' key")
        return False
    
    ft_config = config["fast_train"]
    required_keys = ["vectorizer", "model", "accuracy"]
    for key in required_keys:
        if key not in ft_config:
            print(f"✗ Missing '{key}' in fast_train")
            return False
    
    print("✓ Config structure is valid")
    
    # Test 3: Update accuracy and save
    print("\n3. Testing config update...")
    original_accuracy = ft_config.get("accuracy", {}).get("test_accuracy")
    print(f"   Original accuracy: {original_accuracy}")
    
    # Update just the accuracy
    test_accuracy = 0.9876
    config["fast_train"]["accuracy"]["test_accuracy"] = test_accuracy
    config["fast_train"]["last_updated"] = "2025-12-11T18:00:00Z"
    
    try:
        save_model_config(config)
        print("✓ Config saved successfully")
    except Exception as e:
        print(f"✗ Failed to save config: {e}")
        return False
    
    # Test 4: Reload and verify update
    print("\n4. Verifying saved changes...")
    try:
        reloaded_config = load_model_config()
        new_accuracy = reloaded_config["fast_train"]["accuracy"]["test_accuracy"]
        if new_accuracy == test_accuracy:
            print(f"✓ Accuracy correctly updated to {new_accuracy}")
        else:
            print(f"✗ Accuracy mismatch: expected {test_accuracy}, got {new_accuracy}")
            return False
    except Exception as e:
        print(f"✗ Failed to reload config: {e}")
        return False
    
    # Restore original value
    reloaded_config["fast_train"]["accuracy"]["test_accuracy"] = original_accuracy
    save_model_config(reloaded_config)
    print(f"   Restored original accuracy: {original_accuracy}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_config_operations()
    sys.exit(0 if success else 1)
