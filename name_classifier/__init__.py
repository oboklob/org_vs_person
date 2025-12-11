"""Name Classifier package for distinguishing between organizations and individuals."""

__version__ = "0.1.0"

from name_classifier.classifier import NameClassifier, classify

__all__ = ["NameClassifier", "classify"]
