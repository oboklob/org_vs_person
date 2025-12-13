"""Main classifier module for name type classification."""
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import joblib
import numpy as np

from name_classifier.config import MODEL_PATH, VECTORIZER_PATH, METADATA_PATH


@dataclass
class ClassificationResult:
    """Result of classification with diagnostics.
    
    Attributes:
        label: Predicted label ("PER" or "ORG")
        p_org: Probability of organization class (0.0 to 1.0)
        reason_codes: Dictionary containing diagnostic information
    """
    label: str
    p_org: float
    reason_codes: Dict[str, any]


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
    
    def classify_with_diagnostics(
        self, 
        name: str,
        use_tier_a_shortcut: bool = False
    ) -> ClassificationResult:
        """Classify a name with diagnostic information.
        
        Args:
            name: The name to classify
            use_tier_a_shortcut: If True, automatically classify Tier A legal forms as ORG
                                (disabled by default for safety)
        
        Returns:
            ClassificationResult with label, probability, and reason codes
            
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
        
        # Initialize ISO matcher for diagnostics
        from name_classifier.iso20275_matcher import ISO20275Matcher
        iso_matcher = ISO20275Matcher()
        
        # Check for legal form match
        suffix_match = iso_matcher.match_legal_form(name)
        
        # Optional Tier A shortcut
        if use_tier_a_shortcut and suffix_match and suffix_match.metadata.tier == 'A':
            return ClassificationResult(
                label="ORG",
                p_org=0.99,
                reason_codes={
                    "matched_legal_form": suffix_match.suffix,
                    "legal_form_tier": "A",
                    "is_ambiguous_short": False,
                    "shortcut_applied": True
                }
            )
        
        # Get prediction and probability
        X = self._vectorizer.transform([name])
        prediction = self._model.predict(X)[0]
        
        # Get probability if model supports it
        if hasattr(self._model, 'predict_proba'):
            proba = self._model.predict_proba(X)[0]
            # Assuming ORG is index 1, PER is index 0 (or vice versa)
            # Need to check which class is which
            classes = self._model.classes_
            org_idx = np.where(classes == 'ORG')[0][0] if 'ORG' in classes else 1
            p_org = float(proba[org_idx])
        else:
            # Model doesn't support probabilities, use binary
            p_org = 1.0 if prediction == 'ORG' else 0.0
        
        # Build reason codes
        reason_codes = {
            "matched_legal_form": suffix_match.suffix if suffix_match else None,
            "legal_form_tier": suffix_match.metadata.tier if suffix_match else None,
            "is_ambiguous_short": suffix_match.is_ambiguous_short if suffix_match else False,
            "shortcut_applied": False
        }
        
        # Add top features if model has coefficients
        if hasattr(self._model, 'coef_'):
            try:
                # Get feature importances
                coef = self._model.coef_[0] if len(self._model.coef_.shape) > 1 else self._model.coef_
                
                # Get engineered feature names (last 32 features)
                from name_classifier.feature_engineering import get_feature_names
                feature_names = get_feature_names()
                
                # Get the engineered feature values
                X_dense = X.toarray() if hasattr(X, 'toarray') else X
                eng_features = X_dense[0, -32:]  # Last 32 are engineered
                eng_coef = coef[-32:]  # Last 32 coefficients
                
                # Calculate contribution scores
                contributions = eng_features * eng_coef
                
                # Get top 5 absolute contributions
                top_indices = np.argsort(np.abs(contributions))[-5:][::-1]
                top_features = [
                    (feature_names[i], float(eng_features[i]), float(contributions[i]))
                    for i in top_indices
                    if eng_features[i] != 0  # Only non-zero features
                ]
                
                reason_codes["top_features"] = top_features[:5]
            except Exception as e:
                # If feature extraction fails, don't include it
                reason_codes["top_features_error"] = str(e)
        
        return ClassificationResult(
            label=prediction,
            p_org=p_org,
            reason_codes=reason_codes
        )
    
    def classify_list_with_diagnostics(
        self,
        names: List[str],
        use_tier_a_shortcut: bool = False
    ) -> List[ClassificationResult]:
        """Classify a list of names with diagnostics.
        
        Args:
            names: List of names to classify
            use_tier_a_shortcut: If True, automatically classify Tier A legal forms as ORG
            
        Returns:
            List of ClassificationResults with diagnostics
            
        Raises:
            ValueError: If names is invalid
        """
        # Input validation
        if names is None:
            raise ValueError("Names list cannot be None")
        
        if not isinstance(names, list):
            raise ValueError("Names must be a list")
        
        if not names:
            raise ValueError("Names list cannot be empty")
        
        # Classify each name (could be optimized for batch processing)
        results = []
        for name in names:
            result = self.classify_with_diagnostics(name, use_tier_a_shortcut)
            results.append(result)
        
        return results


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
