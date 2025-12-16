"""Tests for config.py coverage"""
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml

from name_classifier.config import load_model_config, save_model_config, MODEL_CONFIG_PATH


class TestLoadModelConfig:
    """Tests for load_model_config function."""

    def test_file_not_found_error(self):
        """Test lines 37-41: FileNotFoundError when config doesn't exist."""
        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Model configuration not found"):
                load_model_config()

    def test_empty_config_error(self):
        """Test lines 46-47: ValueError when config is empty."""
        with patch.object(Path, 'exists', return_value=True):
            with patch("builtins.open", mock_open(read_data="")):
                with pytest.raises(ValueError, match="Config file .* is empty or invalid"):
                    load_model_config()

    def test_successful_load(self):
        """Test successful config loading."""
        test_config = {
            "fast_train": {
                "vectorizer": {"type": "TfidfVectorizer"},
                "model": {"type": "LogisticRegression"}
            }
        }
        
        with patch.object(Path, 'exists', return_value=True):
            with patch("builtins.open", mock_open(read_data=yaml.dump(test_config))):
                config = load_model_config()
                assert config == test_config


class TestSaveModelConfig:
    """Tests for save_model_config function."""

    @patch('pathlib.Path.mkdir')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_config_creates_directory(self, mock_file, mock_mkdir):
        """Test lines 59-62: save_model_config creates directory and saves YAML."""
        test_config = {
            "fast_train": {
                "vectorizer": {"type": "TfidfVectorizer"},
                "model": {"type": "LogisticRegression"},
                "accuracy": {"test_accuracy": 0.95}
            }
        }
        
        save_model_config(test_config)
        
        # Verify directory creation was attempted
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # Verify file was opened for writing
        mock_file.assert_called_once()
        
        # Verify YAML was written (check that write was called with YAML content)
        handle = mock_file()
        written_data = ''.join(call.args[0] for call in handle.write.call_args_list)
        loaded_config = yaml.safe_load(written_data)
        assert loaded_config == test_config
