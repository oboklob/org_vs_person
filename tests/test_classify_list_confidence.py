"""Unit tests for classify_list with min_confidence."""
import pytest
from unittest.mock import Mock, patch
import numpy as np

from org_vs_person.classifier import NameClassifier, classify_list

class TestClassifyListConfidence:
    """Tests for classify_list with min_confidence."""

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classify_list_no_confidence_arg(self, mock_exists, mock_load):
        """Test classify_list without min_confidence (backward compatibility)."""
        mock_exists.return_value = True

        mock_model = Mock()
        mock_model.predict.return_value = np.array(["PER", "ORG"])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1], [0.2]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        results = classifier.classify_list(["Bob", "Google"])
        
        assert results == ["PER", "ORG"]
        # Should call predict, not predict_proba
        mock_model.predict.assert_called_once()
        mock_model.predict_proba.assert_not_called()

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classify_list_with_confidence_all_confident(self, mock_exists, mock_load):
        """Test classify_list with min_confidence where all are confident."""
        mock_exists.return_value = True

        mock_model = Mock()
        # predict_proba returns [PER, ORG] probabilities
        # Bob: 0.95 PER -> conf = abs(0.05 - 0.5) * 2 = 0.9
        # Google: 0.95 ORG -> conf = abs(0.95 - 0.5) * 2 = 0.9
        mock_model.predict_proba.return_value = np.array([
            [0.95, 0.05], 
            [0.05, 0.95]
        ])
        mock_model.classes_ = np.array(["PER", "ORG"])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1], [0.2]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        results = classifier.classify_list(["Bob", "Google"], min_confidence=0.8)
        
        assert results == ["PER", "ORG"]
        mock_model.predict_proba.assert_called_once()

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classify_list_with_confidence_mixed(self, mock_exists, mock_load):
        """Test classify_list with min_confidence where some are UNK."""
        mock_exists.return_value = True

        mock_model = Mock()
        # Bob: 0.95 PER -> conf 0.9 (Pass)
        # Unknown: 0.55 ORG -> conf 0.1 (Fail)
        # Google: 0.95 ORG -> conf 0.9 (Pass)
        mock_model.predict_proba.return_value = np.array([
            [0.95, 0.05],
            [0.45, 0.55],
            [0.05, 0.95]
        ])
        mock_model.classes_ = np.array(["PER", "ORG"])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1], [0.2], [0.3]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        results = classifier.classify_list(["Bob", "Unknown", "Google"], min_confidence=0.8)
        
        assert results == ["PER", "UNK", "ORG"]

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_module_level_function(self, mock_exists, mock_load):
        """Test module level classify_list function passes min_confidence."""
        mock_exists.return_value = True
        
        # Reset singleton
        import org_vs_person.classifier as clf_module
        clf_module._default_classifier = None

        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.45, 0.55]])
        mock_model.classes_ = np.array(["PER", "ORG"])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        # Should return UNK because confidence is low (0.1)
        results = classify_list(["Unknown"], min_confidence=0.8)
        assert results == ["UNK"]
