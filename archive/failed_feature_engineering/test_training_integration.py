"""Integration tests for enhanced training pipeline."""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from name_classifier.transformers import NameFeatureExtractor, TextExtractor
from name_classifier.feature_dropout import apply_feature_dropout, extract_tier_metadata
from name_classifier.iso20275_matcher import ISO20275Matcher
from name_classifier.feature_engineering import extract_features


class TestNameFeatureExtractor:
    """Test NameFeatureExtractor transformer."""
    
    def test_fit_transform(self):
        """Test basic fit and transform."""
        extractor = NameFeatureExtractor()
        X = ["Company Ltd", "John Smith", "Ministry of Defense"]
        
        extractor.fit(X)
        features = extractor.transform(X)
        
        assert features.shape == (3, 32)
        assert features.dtype == np.float32
    
    def test_with_dataframe(self):
        """Test with DataFrame input."""
        extractor = NameFeatureExtractor()
        df = pd.DataFrame({
            'label': ["Company Ltd", "John Smith"],
            'language': ['en', 'en']
        })
        
        extractor.fit(df)
        features = extractor.transform(df)
        
        assert features.shape == (2, 32)
    
    def test_get_feature_names(self):
        """Test feature name extraction."""
        extractor = NameFeatureExtractor()
        X = ["Test"]
        extractor.fit(X)
        
        names = extractor.get_feature_names_out()
        assert len(names) == 32
        assert 'has_legal_form' in names


class TestTextExtractor:
    """Test TextExtractor transformer."""
    
    def test_extract_from_array(self):
        """Test extraction from array."""
        extractor = TextExtractor()
        X = ["Test 1", "Test 2"]
        
        result = extractor.transform(X)
        np.testing.assert_array_equal(result,  X)
    
    def test_extract_from_dataframe(self):
        """Test extraction from DataFrame."""
        extractor = TextExtractor(column='label')
        df = pd.DataFrame({
            'label': ["Test 1", "Test 2"],
            'other': [1, 2]
        })
        
        result = extractor.transform(df)
        np.testing.assert_array_equal(result, ["Test 1", "Test 2"])


class TestFeatureDropout:
    """Test feature dropout utility."""
    
    def test_no_dropout_when_no_tier(self):
        """Test that samples without tiers are unchanged."""
        X = np.ones((3, 32))
        tiers = [None, None, None]
        
        X_dropped = apply_feature_dropout(X, tiers, 0.5, 0.5, random_state=42)
        
        # Should be unchanged
        np.testing.assert_array_equal(X_dropped, X)
    
    def test_dropout_applied(self):
        """Test dropout is applied to ISO features."""
        X = np.ones((100, 32))
        tiers = ['A'] * 100  # All Tier A
        
        X_dropped = apply_feature_dropout(X, tiers, 1.0, 1.0, random_state=42)
        
        # All ISO features (0-7) should be zeroed
        assert np.all(X_dropped[:, :8] == 0)
        # Other features should be unchanged
        assert np.all(X_dropped[:, 8:] == 1)
    
    def test_tier_conditional_rates(self):
        """Test different dropout rates for different tiers."""
        np.random.seed(42)
        n_samples = 1000
        X = np.ones((n_samples, 32))
        
        # Half Tier A, half Tier C
        tiers = ['A'] * 500 + ['C'] * 500
        
        X_dropped = apply_feature_dropout(X, tiers, 0.3, 0.5, random_state=42)
        
        # Check dropout rates approximately match
        tier_a_dropout = np.mean(X_dropped[:500, 0] == 0)
        tier_c_dropout = np.mean(X_dropped[500:, 0] == 0)
        
        # Should be roughly 30% and 50%
        assert 0.2 < tier_a_dropout < 0.4
        assert 0.4 < tier_c_dropout < 0.6
    
    def test_extract_tier_metadata(self):
        """Test tier metadata extraction."""
        matcher = ISO20275Matcher()
        names = ["Company Ltd", "John Smith", "Test GmbH"]
        
        tiers = extract_tier_metadata(names, matcher)
        
        assert len(tiers) == 3
        # Some should have tiers (or None)
        assert all(t in ['A', 'B', 'C', None] for t in tiers)


class TestPipelineIntegration:
    """Test full pipeline integration."""
    
    def test_feature_union_pipeline(self):
        """Test that FeatureUnion works with our custom transformers."""
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.pipeline import FeatureUnion, Pipeline
        
        # Create pipeline
        char_ngrams = Pipeline([
            ('text_extract', TextExtractor(column='label')),
            ('hashing', HashingVectorizer(analyzer='char', ngram_range=(3, 5), n_features=2**10))
        ])
        
        engineered = NameFeatureExtractor()
        
        feature_union = FeatureUnion([
            ('char_ngrams', char_ngrams),
            ('engineered', engineered)
        ])
        
        # Test data
        df = pd.DataFrame({
            'label': ["Company Ltd", "John Smith", "Ministry of Defense"]
        })
        
        # Fit and transform
        features = feature_union.fit_transform(df)
        
        # Should have hash features + 32 engineered features
        assert features.shape[0] == 3
        assert features.shape[1] == 2**10 + 32
    
    def test_train_simple_model(self):
        """Test training a simple model with the pipeline."""
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.pipeline import FeatureUnion, Pipeline
        
        # Create small dataset
        X_df = pd.DataFrame({
            'label': [
                "Acme Corporation Ltd",
                "Global Industries Inc",
                "Tech Company GmbH",
                "John Smith",
                "Mary Johnson",
                "Bob Williams"
            ]
        })
        y = np.array(['ORG', 'ORG', 'ORG', 'PER', 'PER', 'PER'])
        
        # Create pipeline
        char_ngrams = Pipeline([
            ('text_extract', TextExtractor()),
            ('hashing', HashingVectorizer(analyzer='char', ngram_range=(2, 3), n_features=2**8))
        ])
        
        feature_union = FeatureUnion([
            ('char_ngrams', char_ngrams),
            ('engineered', NameFeatureExtractor())
        ])
        
        X = feature_union.fit_transform(X_df)
        
        # Train simple model
        model = SGDClassifier(loss='log_loss', random_state=42)
        model.fit(X, y)
        
        # Test prediction
        predictions = model.predict(X)
        
        # Should get reasonable accuracy on training data
        accuracy = (predictions == y).mean()
        assert accuracy > 0.5  # At least better than random
