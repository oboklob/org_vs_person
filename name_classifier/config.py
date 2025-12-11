"""Configuration settings for the name classifier package."""
import os
from pathlib import Path

# Package directory
PACKAGE_DIR = Path(__file__).parent

# Model artifacts directory
MODELS_DIR = PACKAGE_DIR / "models"

# Model file paths
MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Data directory (for training scripts)
DATA_DIR = Path(os.environ.get("NAME_CLASSIFIER_DATA_DIR", PACKAGE_DIR.parent / "data"))

# Default parameters
DEFAULT_MAX_FEATURES = 10000
DEFAULT_NGRAM_RANGE = (2, 4)  # Character n-grams
