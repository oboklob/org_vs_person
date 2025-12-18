"""Additional tests for missing coverage in classifier.py"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np

from org_vs_person.classifier import NameClassifier, ClassificationResult


class TestMissingCoverageClassifier:
    """Tests for uncovered lines in classifier.py"""

    @patch("builtins.open", create=True)
    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_metadata_loading_exception_handling(self, mock_exists, mock_load, mock_open):
        """Test lines 69-71: Metadata loading exception handling."""
        mock_exists.return_value = True
        
        # Mock model and vectorizer
        mock_model = Mock()
        mock_model.predict.return_value = np.array(["PER"])
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1]]
        mock_load.side_effect = [mock_model, mock_vectorizer]
        
        # Make open() raise an exception when reading metadata
        mock_open.side_effect = IOError("Cannot read file")
        
        # Should not crash, just skip metadata
        classifier = NameClassifier()
        result = classifier.classify("Test Name")
        assert result == "PER"
        assert classifier._metadata is None

    def test_classify_with_diagnostics_none_raises_error(self):
        """Test line 235: ValueError for None name in classify_with_diagnostics."""
        classifier = NameClassifier()
        
        with pytest.raises(ValueError, match="Name cannot be None"):
            classifier.classify_with_diagnostics(None)

    def test_classify_with_diagnostics_empty_raises_error(self):
        """Test line 239: ValueError for empty name in classify_with_diagnostics."""
        classifier = NameClassifier()
        
        with pytest.raises(ValueError, match="Name cannot be empty"):
            classifier.classify_with_diagnostics("")
        
        with pytest.raises(ValueError, match="Name cannot be empty"):
            classifier.classify_with_diagnostics("   ")

    @patch("org_vs_person.iso20275_matcher.ISO20275Matcher")
    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_tier_a_shortcut(self, mock_exists, mock_load, mock_iso_class):
        """Test line 253: Tier A shortcut in classify_with_diagnostics."""
        mock_exists.return_value = True
        
        # Mock a Tier A legal form match
        mock_suffix_match = Mock()
        mock_suffix_match.suffix = "ltd"
        mock_suffix_match.metadata.tier = "A"
        mock_suffix_match.is_ambiguous_short = False
        
        mock_iso = Mock()
        mock_iso.match_legal_form.return_value = mock_suffix_match
        mock_iso_class.return_value = mock_iso
        
        # Mock model (shouldn't be called due to shortcut)
        mock_model = Mock()
        mock_vectorizer = Mock()
        mock_load.side_effect = [mock_model, mock_vectorizer]
        
        classifier = NameClassifier()
        result = classifier.classify_with_diagnostics("Acme Ltd", use_tier_a_shortcut=True)
        
        # Should return ORG via shortcut without calling model
        assert result.label == "ORG"
        assert result.p_org == 0.99
        assert result.reason_codes["matched_legal_form"] == "ltd"
        assert result.reason_codes["legal_form_tier"] == "A"
        assert result.reason_codes["shortcut_applied"] is True
        
        # Model should not have been called
        mock_model.predict.assert_not_called()

    @patch("org_vs_person.iso20275_matcher.ISO20275Matcher")
    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_model_without_predict_proba(self, mock_exists, mock_load, mock_iso_class):
        """Test line 278: Probability fallback for models without predict_proba."""
        mock_exists.return_value = True
        
        # Mock ISO matcher
        mock_iso = Mock()
        mock_iso.match_legal_form.return_value = None
        mock_iso_class.return_value = mock_iso
        
        # Mock model WITHOUT predict_proba
        mock_model = Mock(spec=['predict', 'classes_'])
        mock_model.predict.return_value = np.array(["ORG"])
        mock_model.classes_ = np.array(["PER", "ORG"])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1]]
        mock_load.side_effect = [mock_model, mock_vectorizer]
        
        classifier = NameClassifier()
        result = classifier.classify_with_diagnostics("Test Company")
        
        # Should use binary probability (1.0 for ORG, 0.0 for PER)
        assert result.label == "ORG"
        assert result.p_org == 1.0

    def test_classify_list_with_diagnostics_none_raises_error(self):
        """Test line 344: ValueError for None names list."""
        classifier = NameClassifier()
        
        with pytest.raises(ValueError, match="Names list cannot be None"):
            classifier.classify_list_with_diagnostics(None)

    def test_classify_list_with_diagnostics_not_list_raises_error(self):
        """Test line 347: ValueError for non-list input."""
        classifier = NameClassifier()
        
        with pytest.raises(ValueError, match="Names must be a list"):
            classifier.classify_list_with_diagnostics("not a list")

    def test_classify_list_with_diagnostics_empty_raises_error(self):
        """Test line 350: ValueError for empty list."""
        classifier = NameClassifier()
        
        with pytest.raises(ValueError, match="Names list cannot be empty"):
            classifier.classify_list_with_diagnostics([])
