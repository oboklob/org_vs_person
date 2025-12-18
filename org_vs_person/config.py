"""Configuration settings for the name classifier package."""
import os
from pathlib import Path
import yaml
from typing import Dict, Any

# Package directory
PACKAGE_DIR = Path(__file__).parent

# Model artifacts directory
MODELS_DIR = PACKAGE_DIR / "models"

# Model file paths
MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.yaml"

# Data directory (for training scripts)
DATA_DIR = Path(os.environ.get("NAME_CLASSIFIER_DATA_DIR", PACKAGE_DIR.parent / "data"))

# Default parameters
DEFAULT_MAX_FEATURES = 10000
DEFAULT_NGRAM_RANGE = (2, 4)  # Character n-grams


def load_model_config() -> Dict[str, Any]:
    """Load model configuration from YAML file.
    
    Returns:
        Dictionary containing model configuration.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
    """
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Model configuration not found at {MODEL_CONFIG_PATH}. "
            f"Please ensure the config file exists."
        )
    
    with open(MODEL_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config file {MODEL_CONFIG_PATH} is empty or invalid")
    
    return config


def save_model_config(config_data: Dict[str, Any]) -> None:
    """Save model configuration to YAML file.
    
    Args:
        config_data: Dictionary containing model configuration to save.
    """
    # Ensure the models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(MODEL_CONFIG_PATH, 'w') as f:
        yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

