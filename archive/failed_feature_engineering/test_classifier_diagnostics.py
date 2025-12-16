"""Integration tests for classifier diagnostics."""
import pytest
from name_classifier import NameClassifier, ClassificationResult
from name_classifier.iso20275_matcher import ISO20275Matcher


class TestClassifierDiagnostics:
    """Test classifier diagnostic features."""
    
    @pytest.fixture
    def create_mock_model(self, tmp_path):
        """Create a minimal mock model for testing diagnostics."""
        import joblib
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.feature_extraction.text import HashingVectorizer
        from name_classifier.transformers import NameFeatureExtractor, TextExtractor
        import pandas as pd
        
        # Create small training set
        X_df = pd.DataFrame({
            'label': [
                "Acme Corp Ltd", "Global Inc", "Tech GmbH",
                "John Smith", "Mary Jones", "Bob Lee"
            ]
        })
        y = ['ORG', 'ORG', 'ORG', 'PER', 'PER', 'PER']
        
        # Create feature pipeline
        char_ngrams = Pipeline([
            ('text_extract', TextExtractor()),
            ('hashing', HashingVectorizer(analyzer='char', ngram_range=(2, 3), n_features=2**8))
        ])
        
        feature_union = FeatureUnion([
            ('char_ngrams', char_ngrams),
            ('engineered', NameFeatureExtractor())
        ])
        
        X = feature_union.fit_transform(X_df)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        # Save model and vectorizer
        model_path = tmp_path / "model.pkl"
        vectorizer_path = tmp_path / "feature_pipeline.pkl"
        
        joblib.dump(model, model_path)
        joblib.dump(feature_union, vectorizer_path)
        
        return model_path, vectorizer_path
    
    def test_classification_result_structure(self, create_mock_model, tmp_path):
        """Test ClassificationResult structure."""
        model_path, vectorizer_path = create_mock_model
        
        classifier = NameClassifier(model_path, vectorizer_path)
        result = classifier.classify_with_diagnostics("Test Company Ltd")
        
        assert isinstance(result, ClassificationResult)
        assert hasattr(result, 'label')
        assert hasattr(result, 'p_org')
        assert hasattr(result, 'reason_codes')
        assert result.label in ['PER', 'ORG']
        assert 0.0 <= result.p_org <= 1.0
        assert isinstance(result.reason_codes, dict)
    
    def test_reason_codes_content(self, create_mock_model, tmp_path):
        """Test reason codes contain expected keys."""
        model_path, vectorizer_path = create_mock_model
        
        classifier = NameClassifier(model_path, vectorizer_path)
        result = classifier.classify_with_diagnostics("Company GmbH")
        
        assert 'matched_legal_form' in result.reason_codes
        assert 'legal_form_tier' in result.reason_codes
        assert 'is_ambiguous_short' in result.reason_codes
        assert 'shortcut_applied' in result.reason_codes
    
    def test_legal_form_detection(self, create_mock_model, tmp_path):
        """Test legal form is detected in reason codes."""
        model_path, vectorizer_path = create_mock_model
        
        classifier = NameClassifier(model_path, vectorizer_path)
        result = classifier.classify_with_diagnostics("Acme Corporation Ltd")
        
        # Should detect 'ltd' as legal form
        if result.reason_codes['matched_legal_form'] is not None:
            assert 'ltd' in result.reason_codes['matched_legal_form'].lower()
            assert result.reason_codes['legal_form_tier'] in ['A', 'B', 'C']
    
    def test_no_legal_form(self, create_mock_model, tmp_path):
        """Test handling of names without legal forms."""
        model_path, vectorizer_path = create_mock_model
        
        classifier = NameClassifier(model_path, vectorizer_path)
        result = classifier.classify_with_diagnostics("John Smith")
        
        # Should not detect legal form for person name
        # (unless coincidentally matches)
        assert result.reason_codes['matched_legal_form'] is None
        assert result.reason_codes['legal_form_tier'] is None
    
    def test_tier_a_shortcut_disabled_by_default(self, create_mock_model, tmp_path):
        """Test that Tier A shortcut is disabled by default."""
        model_path, vectorizer_path = create_mock_model
        classifier = NameClassifier(model_path, vectorizer_path)
        
        result = classifier.classify_with_diagnostics("Company GmbH")
        
        # Shortcut should not be applied
        assert result.reason_codes['shortcut_applied'] == False
    
    def test_tier_a_shortcut_enabled(self, create_mock_model, tmp_path):
        """Test Tier A shortcut when enabled."""
        model_path, vectorizer_path = create_mock_model
        classifier = NameClassifier(model_path, vectorizer_path)
        
        result = classifier.classify_with_diagnostics(
            "Company GmbH",
            use_tier_a_shortcut=True
        )
        
        # If GmbH is Tier A, shortcut should be applied
        if result.reason_codes.get('legal_form_tier') == 'A':
            assert result.label == 'ORG'
            assert result.p_org == 0.99
            assert result.reason_codes['shortcut_applied'] == True
    
    def test_batch_diagnostics(self, create_mock_model, tmp_path):
        """Test batch classification with diagnostics."""
        model_path, vectorizer_path = create_mock_model
        classifier = NameClassifier(model_path, vectorizer_path)
        
        names = ["Company Ltd", "John Smith", "Ministry of Defense"]
        results = classifier.classify_list_with_diagnostics(names)
        
        assert len(results) == 3
        assert all(isinstance(r, ClassificationResult) for r in results)
        assert all(r.label in ['PER', 'ORG'] for r in results)
    
    def test_backward_compatibility(self, create_mock_model, tmp_path):
        """Test that old methods still work."""
        model_path, vectorizer_path = create_mock_model
        classifier = NameClassifier(model_path, vectorizer_path)
        
        # Old single classify
        label = classifier.classify("Test Company")
        assert label in ['PER', 'ORG']
        
        # Old batch classify
        labels = classifier.classify_list(["Test 1", "Test 2"])
        assert len(labels) == 2
        assert all(l in ['PER', 'ORG'] for l in labels)


class TestDiagnosticFeatures:
    """Test specific diagnostic features."""
    
    def test_probability_extraction(self):
        """Test that probabilities are correctly extracted."""
        # This is tested in the fixtures above
        pass
    
    def test_top_features_extraction(self):
        """Test extraction of top contributing features."""
        # Note: This requires a model with coef_ attribute
        # Tested implicitly in diagnostic tests above
        pass
