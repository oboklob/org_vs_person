"""Main classifier module for name type classification."""
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import joblib
import numpy as np

from org_vs_person.config import MODEL_PATH, VECTORIZER_PATH, METADATA_PATH


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
        
        This is the recommended method for general-purpose classification.
        Uses ML model only for maximum speed (~57,000 names/sec, 96.34% precision).
        
        For higher precision (99.97%) with explainability, use 
        classify_with_diagnostics() with use_tier_a_shortcut=True.

        Args:
            name: The name to classify

        Returns:
            "PER" for person names, "ORG" for organization names

        Raises:
            ValueError: If name is None or empty
            
        Example:
            >>> classifier = NameClassifier()
            >>> classifier.classify("Microsoft Corporation")
            'ORG'
            >>> classifier.classify("John Smith")
            'PER'
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
        
        This is the recommended method for batch classification.
        Uses ML model only for maximum speed (~57,000 names/sec, 96.34% precision).
        Optimized for batch processing with vectorization.
        
        For higher precision (99.97%) with explainability, use 
        classify_list_with_diagnostics() with use_tier_a_shortcut=True.

        Args:
            names: List of names to classify

        Returns:
            List of classifications ("PER" or "ORG") corresponding to each input name

        Raises:
            ValueError: If names is None, not a list, empty, or contains None/empty values

        Example:
            >>> classifier = NameClassifier()
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
        """Classify a name with diagnostic information and optional high-precision mode.
        
        This method provides explainability through legal form detection and feature analysis.
        Note: Slower than classify() due to additional diagnostics (~50% slower).
        
        Performance characteristics:
        - use_tier_a_shortcut=False: 96.34% precision (same as classify())
        - use_tier_a_shortcut=True: 99.97% precision for Tier A detections
        
        Use this when:
        - You need explainability (legal form detection, feature contributions)
        - You want higher precision for organization detection (use_tier_a_shortcut=True)
        - You're willing to trade speed for diagnostic information
        
        Args:
            name: The name to classify
            use_tier_a_shortcut: If True, automatically classify Tier A legal forms as ORG
                                with 99.97% precision (recommended for high-precision workflows)
        
        Returns:
            ClassificationResult with label, probability, and reason codes
            
        Raises:
            ValueError: If name is None or empty
            
        Example:
            >>> classifier = NameClassifier()
            >>> result = classifier.classify_with_diagnostics("Acme Ltd", use_tier_a_shortcut=True)
            >>> result.label
            'ORG'
            >>> result.reason_codes['matched_legal_form']
            'ltd'
            >>> result.reason_codes['legal_form_tier']
            'A'
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
        from org_vs_person.iso20275_matcher import ISO20275Matcher
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
    
    def classify_with_confidence(
        self,
        name: str,
        min_confidence: float = 0.7
    ) -> Tuple[str, float]:
        """Classify a name with confidence threshold.
        
        Returns "UNCERTAIN" if the model's confidence is below the threshold.
        Confidence is calculated as abs(p_org - 0.5) * 2, which converts the
        probability (0-1) to a confidence scale where 0.5 probability = 0 confidence
        and 0.0/1.0 probability = 1.0 confidence.
        
        Args:
            name: The name to classify
            min_confidence: Minimum confidence threshold (0.0-1.0). Default 0.7
        
        Returns:
            Tuple of (label, confidence) where label is "PER", "ORG", or "UNCERTAIN"
            and confidence is a float 0.0-1.0
        
        Raises:
            ValueError: If name is None or empty, or min_confidence is out of range
        
        Example:
            >>> classifier.classify_with_confidence("Bob Smith", min_confidence=0.8)
            ('PER', 0.95)
            >>> classifier.classify_with_confidence("Jordan", min_confidence=0.8)
            ('UNCERTAIN', 0.45)
        """
        # Validate min_confidence
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        
        # Get classification with diagnostics
        result = self.classify_with_diagnostics(name)
        
        # Calculate confidence: distance from 0.5 normalized to 0-1
        confidence = abs(result.p_org - 0.5) * 2
        
        # Check if confidence meets threshold
        if confidence < min_confidence:
            return ("UNCERTAIN", confidence)
        
        return (result.label, confidence)
    
    def classify_list_with_confidence(
        self,
        names: List[str],
        min_confidence: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Classify a list of names with confidence thresholds.
        
        Returns "UNCERTAIN" for names where the model's confidence is below the threshold.
        
        Args:
            names: List of names to classify
            min_confidence: Minimum confidence threshold (0.0-1.0). Default 0.7
        
        Returns:
            List of (label, confidence) tuples where label is "PER", "ORG", or "UNCERTAIN"
        
        Raises:
            ValueError: If names is invalid or min_confidence is out of range
        
        Example:
            >>> classifier.classify_list_with_confidence(['Bob Smith', 'Google Inc.'])
            [('PER', 0.95), ('ORG', 0.88)]
        """
        # Validate min_confidence
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        
        # Get all classifications with diagnostics (batch)
        results = self.classify_list_with_diagnostics(names)
        
        # Convert to confidence format
        output = []
        for result in results:
            confidence = abs(result.p_org - 0.5) * 2
            if confidence < min_confidence:
                output.append(("UNCERTAIN", confidence))
            else:
                output.append((result.label, confidence))
        
        return output
    
    def filter_by_confidence(
        self,
        names: List[str],
        target_label: str,
        min_confidence: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Filter names to only those matching target label with sufficient confidence.
        
        This is useful when you want to extract only names that the model is confident
        about for a specific class (e.g., "give me all names that are definitely persons").
        
        Args:
            names: List of names to classify and filter
            target_label: The label to filter for ("PER" or "ORG")
            min_confidence: Minimum confidence threshold (0.0-1.0). Default 0.7
        
        Returns:
            List of (name, confidence) tuples for names matching target_label
            with confidence >= min_confidence
        
        Raises:
            ValueError: If target_label is invalid or min_confidence is out of range
        
        Example:
            >>> names = ['Bob Smith', 'Google Inc.', 'Jane Doe', 'Apple Inc']
            >>> classifier.filter_by_confidence(names, "PER", min_confidence=0.8)
            [('Bob Smith', 0.95), ('Jane Doe', 0.92)]
        """
        # Validate target_label
        if target_label not in ["PER", "ORG"]:
            raise ValueError("target_label must be 'PER' or 'ORG'")
        
        # Validate min_confidence
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        
        # Get all classifications with confidence
        classifications = self.classify_list_with_confidence(names, min_confidence)
        
        # Filter to target label (excluding UNCERTAIN)
        filtered = []
        for name, (label, confidence) in zip(names, classifications):
            if label == target_label:
                filtered.append((name, confidence))
        
        return filtered


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


def classify_with_confidence(name: str, min_confidence: float = 0.7) -> Tuple[str, float]:
    """
    Classify a name with confidence threshold.

    This is a convenience function that uses a singleton NameClassifier instance.
    Returns "UNCERTAIN" if the model's confidence is below the threshold.

    Args:
        name: The name to classify
        min_confidence: Minimum confidence threshold (0.0-1.0). Default 0.7

    Returns:
        Tuple of (label, confidence) where label is "PER", "ORG", or "UNCERTAIN"

    Example:
        >>> classify_with_confidence("Bob Smith", min_confidence=0.8)
        ('PER', 0.95)
        >>> classify_with_confidence("Jordan", min_confidence=0.8)
        ('UNCERTAIN', 0.45)
    """
    global _default_classifier

    if _default_classifier is None:
        _default_classifier = NameClassifier()

    return _default_classifier.classify_with_confidence(name, min_confidence)


def classify_list_with_confidence(
    names: List[str],
    min_confidence: float = 0.7
) -> List[Tuple[str, float]]:
    """
    Classify a list of names with confidence thresholds.

    This is a convenience function that uses a singleton NameClassifier instance.
    Returns "UNCERTAIN" for names where confidence is below the threshold.

    Args:
        names: List of names to classify
        min_confidence: Minimum confidence threshold (0.0-1.0). Default 0.7

    Returns:
        List of (label, confidence) tuples where label is "PER", "ORG", or "UNCERTAIN"

    Example:
        >>> classify_list_with_confidence(['Bob Smith', 'Google Inc.'])
        [('PER', 0.95), ('ORG', 0.88)]
    """
    global _default_classifier

    if _default_classifier is None:
        _default_classifier = NameClassifier()

    return _default_classifier.classify_list_with_confidence(names, min_confidence)


def filter_by_confidence(
    names: List[str],
    target_label: str,
    min_confidence: float = 0.7
) -> List[Tuple[str, float]]:
    """
    Filter names to only those matching target label with sufficient confidence.

    This is a convenience function that uses a singleton NameClassifier instance.
    Useful for extracting only names the model is confident about.

    Args:
        names: List of names to classify and filter
        target_label: The label to filter for ("PER" or "ORG")
        min_confidence: Minimum confidence threshold (0.0-1.0). Default 0.7

    Returns:
        List of (name, confidence) tuples for names matching target_label

    Example:
        >>> names = ['Bob Smith', 'Google Inc.', 'Jane Doe', 'Apple Inc']
        >>> filter_by_confidence(names, "PER", min_confidence=0.8)
        [('Bob Smith', 0.95), ('Jane Doe', 0.92)]
    """
    global _default_classifier

    if _default_classifier is None:
        _default_classifier = NameClassifier()

    return _default_classifier.filter_by_confidence(names, target_label, min_confidence)
