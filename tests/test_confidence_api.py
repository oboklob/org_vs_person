"""Unit tests for confidence-based classification API."""
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np

from org_vs_person.classifier import (
    NameClassifier,
    ClassificationResult,
    classify_with_confidence,
    classify_list_with_confidence,
    filter_by_confidence,
)


class TestClassifyWithConfidence:
    """Tests for classify_with_confidence method."""

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_high_confidence_person(self, mock_exists, mock_load):
        """Test high confidence classification for person."""
        mock_exists.return_value = True

        # Mock model with high confidence for PER (p_org = 0.05)
        mock_model = Mock()
        mock_model.predict.return_value = np.array(["PER"])
        mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])  # [PER, ORG]
        mock_model.classes_ = np.array(["PER", "ORG"])

        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        label, confidence = classifier.classify_with_confidence("Bob Smith", min_confidence=0.7)

        assert label == "PER"
        assert confidence > 0.7
        # Confidence should be abs(0.05 - 0.5) * 2 = 0.9
        assert abs(confidence - 0.9) < 0.01

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_high_confidence_organization(self, mock_exists, mock_load):
        """Test high confidence classification for organization."""
        mock_exists.return_value = True

        # Mock model with high confidence for ORG (p_org = 0.95)
        mock_model = Mock()
        mock_model.predict.return_value = np.array(["ORG"])
        mock_model.predict_proba.return_value = np.array([[0.05, 0.95]])  # [PER, ORG]
        mock_model.classes_ = np.array(["PER", "ORG"])

        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        label, confidence = classifier.classify_with_confidence("Microsoft Corp", min_confidence=0.7)

        assert label == "ORG"
        assert confidence > 0.7
        # Confidence should be abs(0.95 - 0.5) * 2 = 0.9
        assert abs(confidence - 0.9) < 0.01

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_uncertain_classification(self, mock_exists, mock_load):
        """Test uncertain classification when probability is near 0.5."""
        mock_exists.return_value = True

        # Mock model with low confidence (p_org = 0.55, near 0.5)
        mock_model = Mock()
        mock_model.predict.return_value = np.array(["ORG"])
        mock_model.predict_proba.return_value = np.array([[0.45, 0.55]])  # [PER, ORG]
        mock_model.classes_ = np.array(["PER", "ORG"])

        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        label, confidence = classifier.classify_with_confidence("Jordan", min_confidence=0.7)

        assert label == "UNCERTAIN"
        # Confidence should be abs(0.55 - 0.5) * 2 = 0.1
        assert abs(confidence - 0.1) < 0.01

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_custom_confidence_threshold(self, mock_exists, mock_load):
        """Test custom confidence threshold."""
        mock_exists.return_value = True

        # Mock model with moderate confidence (p_org = 0.75)
        mock_model = Mock()
        mock_model.predict.return_value = np.array(["ORG"])
        mock_model.predict_proba.return_value = np.array([[0.25, 0.75]])  # [PER, ORG]
        mock_model.classes_ = np.array(["PER", "ORG"])

        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1]]

        mock_load.side_effect = [mock_model, mock_vectorizer, mock_model, mock_vectorizer]

        classifier = NameClassifier()

        # Confidence is abs(0.75 - 0.5) * 2 = 0.5
        # Should pass with min_confidence=0.4
        label1, conf1 = classifier.classify_with_confidence("Test Name", min_confidence=0.4)
        assert label1 == "ORG"
        assert abs(conf1 - 0.5) < 0.01

        # Should fail with min_confidence=0.6
        label2, conf2 = classifier.classify_with_confidence("Test Name", min_confidence=0.6)
        assert label2 == "UNCERTAIN"
        assert abs(conf2 - 0.5) < 0.01

    def test_invalid_confidence_threshold(self):
        """Test invalid confidence threshold raises ValueError."""
        classifier = NameClassifier()

        with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
            classifier.classify_with_confidence("Test", min_confidence=-0.1)

        with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
            classifier.classify_with_confidence("Test", min_confidence=1.5)


class TestClassifyListWithConfidence:
    """Tests for classify_list_with_confidence method."""

    @patch("org_vs_person.iso20275_matcher.ISO20275Matcher")
    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_batch_classification(self, mock_exists, mock_load, mock_iso_class):
        """Test batch classification with mixed confidence levels."""
        mock_exists.return_value = True

        # Mock ISO matcher to return None (no legal form match)
        mock_iso = Mock()
        mock_iso.match_legal_form.return_value = None
        mock_iso_class.return_value = mock_iso

        # Mock model with various confidence levels
        # classify_list_with_diagnostics calls predict individually for each name
        mock_model = Mock()
        mock_model.predict.side_effect = [
            np.array(["PER"]),  # First call for "Bob Smith"
            np.array(["ORG"]),  # Second call for "Microsoft Corp"
            np.array(["PER"]),  # Third call for "Jordan"
        ]
        mock_model.predict_proba.side_effect = [
            np.array([[0.95, 0.05]]),  # High confidence PER
            np.array([[0.05, 0.95]]),  # High confidence ORG
            np.array([[0.45, 0.55]]),  # Low confidence (uncertain)
        ]
        mock_model.classes_ = np.array(["PER", "ORG"])

        mock_vectorizer = Mock()
        mock_vectorizer.transform.side_effect = [[[0.1]], [[0.2]], [[0.3]]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        results = classifier.classify_list_with_confidence(
            ["Bob Smith", "Microsoft Corp", "Jordan"],
            min_confidence=0.7
        )

        assert len(results) == 3
        assert results[0][0] == "PER"
        assert results[0][1] > 0.7
        assert results[1][0] == "ORG"
        assert results[1][1] > 0.7
        assert results[2][0] == "UNCERTAIN"
        assert results[2][1] < 0.7

    def test_invalid_confidence_threshold(self):
        """Test invalid confidence threshold raises ValueError."""
        classifier = NameClassifier()

        with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
            classifier.classify_list_with_confidence(["Test"], min_confidence=2.0)


class TestFilterByConfidence:
    """Tests for filter_by_confidence method."""

    @patch("org_vs_person.iso20275_matcher.ISO20275Matcher")
    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_filter_persons(self, mock_exists, mock_load, mock_iso_class):
        """Test filtering to get only confident persons."""
        mock_exists.return_value = True

        # Mock ISO matcher to return None (no legal form match)
        mock_iso = Mock()
        mock_iso.match_legal_form.return_value = None
        mock_iso_class.return_value = mock_iso

        mock_model = Mock()
        # Individual calls for each name
        mock_model.predict.side_effect = [
            np.array(["PER"]),  # Bob Smith
            np.array(["ORG"]),  # Microsoft Corp
            np.array(["PER"]),  # Jordan
            np.array(["ORG"]),  # Apple Inc
        ]
        # Mix of high and low confidence
        mock_model.predict_proba.side_effect = [
            np.array([[0.95, 0.05]]),  # High conf PER - should pass
            np.array([[0.05, 0.95]]),  # High conf ORG - filtered out
            np.array([[0.55, 0.45]]),  # Low conf PER - should fail threshold
            np.array([[0.20, 0.80]]),  # High conf ORG - filtered out
        ]
        mock_model.classes_ = np.array(["PER", "ORG"])

        mock_vectorizer = Mock()
        mock_vectorizer.transform.side_effect = [[[0.1]], [[0.2]], [[0.3]], [[0.4]]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        names = ["Bob Smith", "Microsoft Corp", "Jordan", "Apple Inc"]
        results = classifier.filter_by_confidence(names, "PER", min_confidence=0.7)

        # Only "Bob Smith" should pass (high confidence PER)
        assert len(results) == 1
        assert results[0][0] == "Bob Smith"
        assert results[0][1] > 0.7

    @patch("org_vs_person.iso20275_matcher.ISO20275Matcher")
    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_filter_organizations(self, mock_exists, mock_load, mock_iso_class):
        """Test filtering to get only confident organizations."""
        mock_exists.return_value = True

        # Mock ISO matcher to return None (no legal form match)
        mock_iso = Mock()
        mock_iso.match_legal_form.return_value = None
        mock_iso_class.return_value = mock_iso

        mock_model = Mock()
        # Individual calls for each name
        mock_model.predict.side_effect = [
            np.array(["PER"]),  # Bob Smith
            np.array(["ORG"]),  # Microsoft Corp
            np.array(["PER"]),  # Jordan
            np.array(["ORG"]),  # Apple Inc
        ]
        mock_model.predict_proba.side_effect = [
            np.array([[0.95, 0.05]]),  # High conf PER - filtered out
            np.array([[0.05, 0.95]]),  # High conf ORG - should pass (conf = 0.9)
            np.array([[0.55, 0.45]]),  # Low conf PER - filtered out
            np.array([[0.15, 0.85]]),  # High conf ORG - should pass (conf = 0.7)
        ]
        mock_model.classes_ = np.array(["PER", "ORG"])

        mock_vectorizer = Mock()
        mock_vectorizer.transform.side_effect = [[[0.1]], [[0.2]], [[0.3]], [[0.4]]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        names = ["Bob Smith", "Microsoft Corp", "Jordan", "Apple Inc"]
        results = classifier.filter_by_confidence(names, "ORG", min_confidence=0.7)

        # Should get both high confidence ORGs
        assert len(results) == 2
        assert results[0][0] == "Microsoft Corp"
        assert results[1][0] == "Apple Inc"

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_filter_empty_result(self, mock_exists, mock_load):
        """Test filtering that returns no results."""
        mock_exists.return_value = True

        # All low confidence
        mock_model = Mock()
        mock_model.predict.return_value = np.array(["PER", "PER"])
        mock_model.predict_proba.return_value = np.array([
            [0.55, 0.45],  # Low confidence
            [0.52, 0.48],  # Low confidence
        ])
        mock_model.classes_ = np.array(["PER", "ORG"])

        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1], [0.2]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        names = ["Name1", "Name2"]
        results = classifier.filter_by_confidence(names, "PER", min_confidence=0.9)

        assert results == []

    def test_invalid_target_label(self):
        """Test invalid target label raises ValueError."""
        classifier = NameClassifier()

        with pytest.raises(ValueError, match="target_label must be 'PER' or 'ORG'"):
            classifier.filter_by_confidence(["Test"], "INVALID", min_confidence=0.7)

    def test_invalid_confidence_threshold(self):
        """Test invalid confidence threshold raises ValueError."""
        classifier = NameClassifier()

        with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
            classifier.filter_by_confidence(["Test"], "PER", min_confidence=-0.5)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classify_with_confidence_function(self, mock_exists, mock_load):
        """Test classify_with_confidence convenience function."""
        mock_exists.return_value = True

        mock_model = Mock()
        mock_model.predict.return_value = np.array(["PER"])
        mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])
        mock_model.classes_ = np.array(["PER", "ORG"])

        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        # Reset singleton
        import org_vs_person.classifier as clf_module
        clf_module._default_classifier = None

        label, confidence = classify_with_confidence("Bob Smith", min_confidence=0.7)
        assert label == "PER"
        assert confidence > 0.7

    @patch("org_vs_person.iso20275_matcher.ISO20275Matcher")
    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classify_list_with_confidence_function(self, mock_exists, mock_load, mock_iso_class):
        """Test classify_list_with_confidence convenience function."""
        mock_exists.return_value = True

        # Mock ISO matcher to return None (no legal form match)
        mock_iso = Mock()
        mock_iso.match_legal_form.return_value = None
        mock_iso_class.return_value = mock_iso

        mock_model = Mock()
        # Individual calls for each name
        mock_model.predict.side_effect = [
            np.array(["PER"]),
            np.array(["ORG"]),
        ]
        mock_model.predict_proba.side_effect = [
            np.array([[0.95, 0.05]]),
            np.array([[0.05, 0.95]]),
        ]
        mock_model.classes_ = np.array(["PER", "ORG"])

        mock_vectorizer = Mock()
        mock_vectorizer.transform.side_effect = [[[0.1]], [[0.2]]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        # Reset singleton
        import org_vs_person.classifier as clf_module
        clf_module._default_classifier = None

        results = classify_list_with_confidence(["Bob Smith", "Microsoft Corp"])
        assert len(results) == 2
        assert results[0][0] == "PER"
        assert results[1][0] == "ORG"

    @patch("org_vs_person.iso20275_matcher.ISO20275Matcher")
    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_filter_by_confidence_function(self, mock_exists, mock_load, mock_iso_class):
        """Test filter_by_confidence convenience function."""
        mock_exists.return_value = True

        # Mock ISO matcher to return None (no legal form match)
        mock_iso = Mock()
        mock_iso.match_legal_form.return_value = None
        mock_iso_class.return_value = mock_iso

        mock_model = Mock()
        # Individual calls for each name
        mock_model.predict.side_effect = [
            np.array(["PER"]),
            np.array(["ORG"]),
        ]
        mock_model.predict_proba.side_effect = [
            np.array([[0.95, 0.05]]),
            np.array([[0.05, 0.95]]),
        ]
        mock_model.classes_ = np.array(["PER", "ORG"])

        mock_vectorizer = Mock()
        mock_vectorizer.transform.side_effect = [[[0.1]], [[0.2]]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        # Reset singleton
        import org_vs_person.classifier as clf_module
        clf_module._default_classifier = None

        results = filter_by_confidence(["Bob Smith", "Microsoft Corp"], "PER", min_confidence=0.7)
        assert len(results) == 1
        assert results[0][0] == "Bob Smith"
