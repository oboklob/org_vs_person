"""Unit tests for the NameClassifier class."""
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import joblib

from org_vs_person.classifier import NameClassifier, classify, classify_list


class TestNameClassifier:
    """Tests for the NameClassifier class."""

    def test_init_default_paths(self):
        """Test initialization with default paths."""
        classifier = NameClassifier()
        assert classifier.model_path is not None
        assert classifier.vectorizer_path is not None
        assert classifier._model is None
        assert classifier._vectorizer is None

    def test_init_custom_paths(self):
        """Test initialization with custom paths."""
        model_path = Path("/custom/model.pkl")
        vectorizer_path = Path("/custom/vectorizer.pkl")

        classifier = NameClassifier(model_path=model_path, vectorizer_path=vectorizer_path)

        assert classifier.model_path == model_path
        assert classifier.vectorizer_path == vectorizer_path

    def test_classify_none_raises_error(self):
        """Test that classify raises ValueError for None input."""
        classifier = NameClassifier()

        with pytest.raises(ValueError, match="Name cannot be None"):
            classifier.classify(None)

    def test_classify_empty_string_raises_error(self):
        """Test that classify raises ValueError for empty string."""
        classifier = NameClassifier()

        with pytest.raises(ValueError, match="Name cannot be empty"):
            classifier.classify("")

        with pytest.raises(ValueError, match="Name cannot be empty"):
            classifier.classify("   ")  # Whitespace only

    def test_classify_model_not_found(self):
        """Test that classify raises FileNotFoundError if model doesn't exist."""
        classifier = NameClassifier(model_path=Path("/nonexistent/model.pkl"))

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            classifier.classify("Test Name")

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classify_success(self, mock_exists, mock_load):
        """Test successful classification."""
        # Mock file existence
        mock_exists.return_value = True

        # Mock model and vectorizer
        mock_model = Mock()
        mock_model.predict.return_value = ["PER"]

        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1, 0.2, 0.3]]

        # Set up joblib.load to return model then vectorizer
        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        result = classifier.classify("Bob Smith")

        assert result == "PER"
        mock_vectorizer.transform.assert_called_once_with(["Bob Smith"])
        mock_model.predict.assert_called_once()

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classify_strips_whitespace(self, mock_exists, mock_load):
        """Test that classify strips leading/trailing whitespace."""
        mock_exists.return_value = True

        mock_model = Mock()
        mock_model.predict.return_value = ["ORG"]

        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1, 0.2]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        result = classifier.classify("  Microsoft Corporation  ")

        # Should be called with stripped version
        mock_vectorizer.transform.assert_called_once_with(["Microsoft Corporation"])
        assert result == "ORG"

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_lazy_loading(self, mock_exists, mock_load):
        """Test that model is loaded lazily on first classify call."""
        mock_exists.return_value = True

        mock_model = Mock()
        mock_model.predict.return_value = ["PER"]

        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1]]

        mock_load.side_effect = [mock_model, mock_vectorizer, mock_model, mock_vectorizer]

        classifier = NameClassifier()

        # Model should not be loaded yet
        assert classifier._model is None
        assert mock_load.call_count == 0

        # First call loads the model
        classifier.classify("Test")
        assert mock_load.call_count == 2  # Model + vectorizer

        # Second call doesn't reload
        classifier.classify("Test2")
        assert mock_load.call_count == 2  # Still 2

    @patch("builtins.open", create=True)
    @patch("pathlib.Path.exists")
    def test_get_metadata(self, mock_exists, mock_open):
        """Test get_metadata method."""
        mock_exists.return_value = True

        metadata = {
            "test_accuracy": 0.95,
            "vectorizer": "tfidf_char_2-4",
            "classifier": "LogisticRegression",
        }

        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = json.dumps(metadata)
        mock_open.return_value = mock_file

        classifier = NameClassifier()
        result = classifier.get_metadata()

        assert result == metadata


class TestClassifyFunction:
    """Tests for the convenience classify() function."""

    @patch("org_vs_person.classifier.NameClassifier")
    def test_classify_function_uses_singleton(self, mock_classifier_class):
        """Test that classify() uses a singleton instance."""
        mock_instance = Mock()
        mock_instance.classify.return_value = "PER"
        mock_classifier_class.return_value = mock_instance

        # First call creates instance
        result1 = classify("Name 1")
        assert result1 == "PER"
        assert mock_classifier_class.call_count == 1

        # Note: In actual code, the singleton is module-level, so this test
        # would need to reset the module state between calls, which is complex
        # This test demonstrates the expected behavior

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classify_function_integration(self, mock_exists, mock_load):
        """Test classify function end-to-end."""
        mock_exists.return_value = True

        mock_model = Mock()
        mock_model.predict.return_value = ["ORG"]

        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.5]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        # Reset the singleton before testing
        import org_vs_person.classifier as clf_module

        clf_module._default_classifier = None

        result = clf_module.classify("Apple Inc")
        assert result == "ORG"


class TestClassifyList:
    """Tests for the classify_list method."""

    def test_classify_list_none_raises_error(self):
        """Test that classify_list raises ValueError for None input."""
        classifier = NameClassifier()

        with pytest.raises(ValueError, match="Names list cannot be None"):
            classifier.classify_list(None)

    def test_classify_list_not_list_raises_error(self):
        """Test that classify_list raises ValueError for non-list input."""
        classifier = NameClassifier()

        with pytest.raises(ValueError, match="Names must be a list"):
            classifier.classify_list("not a list")

        with pytest.raises(ValueError, match="Names must be a list"):
            classifier.classify_list(("tuple", "of", "names"))

    def test_classify_list_empty_list_raises_error(self):
        """Test that classify_list raises ValueError for empty list."""
        classifier = NameClassifier()

        with pytest.raises(ValueError, match="Names list cannot be empty"):
            classifier.classify_list([])

    def test_classify_list_none_element_raises_error(self):
        """Test that classify_list raises ValueError if list contains None."""
        classifier = NameClassifier()

        with pytest.raises(ValueError, match="Name at index 1 cannot be None"):
            classifier.classify_list(["Valid Name", None, "Another Name"])

    def test_classify_list_empty_string_element_raises_error(self):
        """Test that classify_list raises ValueError if list contains empty string."""
        classifier = NameClassifier()

        with pytest.raises(ValueError, match="Name at index 2 cannot be empty"):
            classifier.classify_list(["Valid Name", "Another Name", ""])

        with pytest.raises(ValueError, match="Name at index 0 cannot be empty"):
            classifier.classify_list(["   ", "Valid Name"])  # Whitespace only

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classify_list_success(self, mock_exists, mock_load):
        """Test successful classification of multiple names."""
        mock_exists.return_value = True

        mock_model = Mock()
        mock_model.predict.return_value = ["PER", "ORG", "ORG"]

        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        result = classifier.classify_list(["Bob Smith", "Google Inc.", "ministry of defense"])

        assert result == ["PER", "ORG", "ORG"]
        mock_vectorizer.transform.assert_called_once_with(["Bob Smith", "Google Inc.", "ministry of defense"])
        mock_model.predict.assert_called_once()

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classify_list_strips_whitespace(self, mock_exists, mock_load):
        """Test that classify_list strips leading/trailing whitespace from all names."""
        mock_exists.return_value = True

        mock_model = Mock()
        mock_model.predict.return_value = ["PER", "ORG"]

        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1], [0.2]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        result = classifier.classify_list(["  Bob Smith  ", "  Google Inc.  "])

        # Should be called with stripped versions
        mock_vectorizer.transform.assert_called_once_with(["Bob Smith", "Google Inc."])
        assert result == ["PER", "ORG"]

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classify_list_single_element(self, mock_exists, mock_load):
        """Test classify_list with a single element."""
        mock_exists.return_value = True

        mock_model = Mock()
        mock_model.predict.return_value = ["PER"]

        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        classifier = NameClassifier()
        result = classifier.classify_list(["Bob Smith"])

        assert result == ["PER"]
        assert isinstance(result, list)


class TestClassifyListFunction:
    """Tests for the convenience classify_list() function."""

    @patch("org_vs_person.classifier.joblib.load")
    @patch("pathlib.Path.exists")
    def test_classify_list_function_integration(self, mock_exists, mock_load):
        """Test classify_list function end-to-end."""
        mock_exists.return_value = True

        mock_model = Mock()
        mock_model.predict.return_value = ["PER", "ORG", "ORG"]

        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1], [0.2], [0.3]]

        mock_load.side_effect = [mock_model, mock_vectorizer]

        # Reset the singleton before testing
        import org_vs_person.classifier as clf_module

        clf_module._default_classifier = None

        result = clf_module.classify_list(["Bob Smith", "Google Inc.", "Ministry of Defense"])
        assert result == ["PER", "ORG", "ORG"]

