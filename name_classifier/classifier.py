"""Main classifier module for name type classification."""
import json
import warnings
from pathlib import Path
from typing import Optional

import joblib

from name_classifier.config import MODEL_PATH, VECTORIZER_PATH, METADATA_PATH


class NameClassifier:
    """Classifier for determining if a name is a person (PER) or organization (ORG)."""

    def __init__(self, model_path: Optional[Path] = None, vectorizer_path: Optional[Path] = None):
        """
        Initialize the NameClassifier.

        Args:
            model_path: Optional custom path to the trained model file
            vectorizer_path: Optional custom path to the vectorizer file
        """
        self.model_path = model_path or MODEL_PATH
        self.vectorizer_path = vectorizer_path or VECTORIZER_PATH
        self._model = None
        self._vectorizer = None
        self._metadata = None

    def _load_model(self):
        """Lazy load the trained model and vectorizer."""
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {self.model_path}. "
                    "Please train a model first using scripts/train_model.py"
                )

            self._model = joblib.load(self.model_path)

            if not self.vectorizer_path.exists():
                raise FileNotFoundError(
                    f"Vectorizer file not found at {self.vectorizer_path}. "
                    "Please train a model first using scripts/train_model.py"
                )

            self._vectorizer = joblib.load(self.vectorizer_path)

            # Load metadata if available
            try:
                if METADATA_PATH.exists():
                    with open(METADATA_PATH, "r") as f:
                        self._metadata = json.load(f)
            except (FileNotFoundError, IOError):
                # Metadata is optional
                pass

    def classify(self, name: str) -> str:
        """
        Classify a name as either PER (person) or ORG (organization).

        Args:
            name: The name to classify

        Returns:
            "PER" for person names, "ORG" for organization names

        Raises:
            ValueError: If name is None or empty
        """
        # Input validation
        if name is None:
            raise ValueError("Name cannot be None")

        name = str(name).strip()

        if not name:
            raise ValueError("Name cannot be empty")

        # Load model on first use
        self._load_model()

        # Vectorize the name
        X = self._vectorizer.transform([name])

        # Predict
        prediction = self._model.predict(X)[0]

        return prediction

    def classify_list(self, names: list) -> list:
        """
        Classify a list of names as either PER (person) or ORG (organization).

        Args:
            names: List of names to classify

        Returns:
            List of classifications ("PER" or "ORG") corresponding to each input name

        Raises:
            ValueError: If names is None, not a list, empty, or contains None/empty values

        Example:
            >>> classifier.classify_list(['Bob Smith', 'Google Inc.', 'ministry of defense'])
            ['PER', 'ORG', 'ORG']
        """
        # Input validation
        if names is None:
            raise ValueError("Names list cannot be None")

        if not isinstance(names, list):
            raise ValueError("Names must be a list")

        if not names:
            raise ValueError("Names list cannot be empty")

        # Validate and clean each name
        cleaned_names = []
        for i, name in enumerate(names):
            if name is None:
                raise ValueError(f"Name at index {i} cannot be None")

            cleaned_name = str(name).strip()
            if not cleaned_name:
                raise ValueError(f"Name at index {i} cannot be empty")

            cleaned_names.append(cleaned_name)

        # Load model on first use
        self._load_model()

        # Vectorize all names at once (batch processing)
        X = self._vectorizer.transform(cleaned_names)

        # Predict all at once
        predictions = self._model.predict(X)

        # Convert to list if needed (numpy array has tolist(), regular list doesn't)
        if hasattr(predictions, 'tolist'):
            return predictions.tolist()
        return list(predictions)

    def get_metadata(self) -> Optional[dict]:
        """
        Get model metadata if available.

        Returns:
            Dictionary with model metadata (accuracy, training date, etc.) or None
        """
        if self._metadata is None and METADATA_PATH.exists():
            with open(METADATA_PATH, "r") as f:
                self._metadata = json.load(f)

        return self._metadata


# Singleton instance for convenience
_default_classifier = None


def classify(name: str) -> str:
    """
    Classify a name as either PER (person) or ORG (organization).

    This is a convenience function that uses a singleton NameClassifier instance.

    Args:
        name: The name to classify

    Returns:
        "PER" for person names, "ORG" for organization names

    Example:
        >>> classify("Bob Smith")
        'PER'
        >>> classify("Microsoft Corporation")
        'ORG'
    """
    global _default_classifier

    if _default_classifier is None:
        _default_classifier = NameClassifier()

    return _default_classifier.classify(name)


def classify_list(names: list) -> list:
    """
    Classify a list of names as either PER (person) or ORG (organization).

    This is a convenience function that uses a singleton NameClassifier instance.

    Args:
        names: List of names to classify

    Returns:
        List of classifications ("PER" or "ORG") corresponding to each input name

    Example:
        >>> classify_list(['Bob Smith', 'Google Inc.', 'ministry of defense'])
        ['PER', 'ORG', 'ORG']
    """
    global _default_classifier

    if _default_classifier is None:
        _default_classifier = NameClassifier()

    return _default_classifier.classify_list(names)
