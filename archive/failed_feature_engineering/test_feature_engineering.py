"""Unit tests for feature engineering module."""
import pytest
import numpy as np
from name_classifier.feature_engineering import (
    extract_features,
    extract_iso_features,
    extract_language_jurisdiction_features,
    extract_string_structure_features,
    extract_casing_initials_features,
    extract_lexical_features,
    bucket_language,
    get_feature_names
)
from name_classifier.iso20275_matcher import ISO20275Matcher


@pytest.fixture
def iso_matcher():
    """Create ISO20275Matcher instance."""
    return ISO20275Matcher()


class TestBucketLanguage:
    """Tests for language bucketing."""
    
    def test_en_like_bucket(self):
        assert bucket_language('en') == 'en_like'
        assert bucket_language('us') == 'en_like'
        assert bucket_language('au') == 'en_like'
        assert bucket_language('gb') == 'en_like'
    
    def test_germanic_bucket(self):
        assert bucket_language('de') == 'germanic'
        assert bucket_language('nl') == 'germanic'
        assert bucket_language('at') == 'germanic'
    
    def test_romance_bucket(self):
        assert bucket_language('fr') == 'romance'
        assert bucket_language('es') == 'romance'
        assert bucket_language('it') == 'romance'
    
    def test_nordic_bucket(self):
        assert bucket_language('se') == 'nordic'
        assert bucket_language('no') == 'nordic'
        assert bucket_language('dk') == 'nordic'
    
    def test_other_bucket(self):
        assert bucket_language('ja') == 'other'
        assert bucket_language('zh') == 'other'
    
    def test_unknown_bucket(self):
        assert bucket_language(None) == 'unknown'
        assert bucket_language('') == 'unknown'


class TestExtractISOFeatures:
    """Tests for ISO 20275-derived features."""
    
    def test_with_legal_form(self, iso_matcher):
        features = extract_iso_features("Company Ltd", iso_matcher)
        assert features.shape == (8,)
        assert features[0] == 1.0  # has_legal_form
    
    def test_without_legal_form(self, iso_matcher):
        features = extract_iso_features("John Smith", iso_matcher)
        assert features.shape == (8,)
        # May or may not match depending on data
    
    def test_tier_one_hot(self, iso_matcher):
        features = extract_iso_features("Company GmbH", iso_matcher)
        # GmbH should be Tier A (4 chars)
        if features[0] == 1.0:  # has match
            # Exactly one tier should be 1.0
            tier_sum = features[1] + features[2] + features[3]
            assert tier_sum == 1.0
    
    def test_token_char_lengths(self, iso_matcher):
        features = extract_iso_features("Test Ltd", iso_matcher)
        if features[0] == 1.0:  # has match
            assert features[4] >= 1.0  # token_len
            assert features[5] >= 1.0  # char_len


# NOTE: TestExtractLanguageJurisdictionFeatures class removed
# Language/jurisdiction features were unused (all zeros) and have been removed from feature set


class TestExtractStringStructureFeatures:
    """Tests for string structure features."""
    
    def test_basic_structure(self):
        features = extract_string_structure_features("Company Name", "Company Name")
        assert features.shape == (8,)
        assert features[0] > 0  # char_len
        assert features[1] >= 2  # token_count (at least 2)
        assert features[2] > 0  # avg_token_len
    
    def test_contains_digit(self):
        features = extract_string_structure_features("Company123", "Company123")
        assert features[3] == 1.0  # contains_digit
    
    def test_contains_ampersand(self):
        features = extract_string_structure_features("A & B", "A & B")
        assert features[4] == 1.0  # contains_ampersand
    
    def test_contains_comma(self):
        features = extract_string_structure_features("Smith, John", "Smith, John")
        assert features[5] == 1.0  # contains_comma
    
    def test_contains_slash_hyphen(self):
        features = extract_string_structure_features("A-B/C", "A-B/C")
        assert features[6] == 1.0  # contains_slash_or_hyphen
    
    def test_contains_period(self):
        features = extract_string_structure_features("Co. Ltd.", "Co. Ltd.")
        assert features[7] == 1.0  # contains_period
    
    def test_avg_token_length(self):
        features = extract_string_structure_features("ABC DEFGH", "ABC DEFGH")
        # avg = (3 + 5) / 2 = 4.0
        assert 3.5 <= features[2] <= 4.5


class TestExtractCasingInitialsFeatures:
    """Tests for casing and initials features."""
    
    def test_all_caps(self):
        features = extract_casing_initials_features("ACME")
        assert features[0] == 1.0  # all_caps_ratio
    
    def test_mixed_case(self):
        features = extract_casing_initials_features("Acme")
        assert 0.0 < features[0] < 1.0  # partial caps
    
    def test_capitalized_tokens(self):
        features = extract_casing_initials_features("John Smith")
        assert features[1] == 1.0  # all tokens capitalized
    
    def test_initials_pattern_with_periods(self):
        features = extract_casing_initials_features("A. B. Smith")
        assert features[2] == 1.0  # has_initials_pattern
    
    def test_initials_pattern_without_periods(self):
        features = extract_casing_initials_features("A B Smith")
        assert features[2] == 1.0  # has_initials_pattern
    
    def test_person_title(self):
        features = extract_casing_initials_features("Dr Smith")
        assert features[3] == 1.0  # has_person_title
        
        features = extract_casing_initials_features("Mr. Jones")
        assert features[3] == 1.0
    
    def test_person_suffix(self):
        features = extract_casing_initials_features("John Smith Jr")
        assert features[4] == 1.0  # has_person_suffix
        
        features = extract_casing_initials_features("Jane Doe PhD")
        assert features[4] == 1.0
    
    def test_no_person_markers(self):
        features = extract_casing_initials_features("Company Name")
        assert features[3] == 0.0  # no title
        assert features[4] == 0.0  # no suffix


class TestExtractLexicalFeatures:
    """Tests for lexical signal features."""
    
    def test_org_keyword(self):
        features = extract_lexical_features("Foundation for Science")
        assert features[0] == 1.0  # has_org_keyword
        
        features = extract_lexical_features("University of California")
        assert features[0] == 1.0
        
        features = extract_lexical_features("ABC Corporation")
        assert features[0] == 1.0
    
    def test_given_name(self):
        features = extract_lexical_features("John Smith")
        assert features[1] == 1.0  # has_given_name_hit
        
        features = extract_lexical_features("Mary Jones")
        assert features[1] == 1.0
    
    def test_surname_particle(self):
        features = extract_lexical_features("Vincent van Gogh")
        assert features[2] == 1.0  # has_surname_particle
        
        features = extract_lexical_features("Jean de la Tour")
        assert features[2] == 1.0
        
        features = extract_lexical_features("Ludwig von Beethoven")
        assert features[2] == 1.0
    
    def test_no_lexical_signals(self):
        features = extract_lexical_features("Random Text")
        # May have 0s for all if no matches
        assert features.shape == (3,)


class TestExtractFeatures:
    """Tests for complete feature extraction."""
    
    def test_feature_count(self, iso_matcher):
        features = extract_features("Test Company Ltd", iso_matcher)
        assert features.shape == (24,)  # Updated from 32 to 24 (removed 8 language features)
    
    def test_feature_types(self, iso_matcher):
        features = extract_features("Test Company", iso_matcher)
        assert features.dtype == np.float32
    
    def test_with_language_hint(self, iso_matcher):
        # Language hints are currently unused but parameter is kept for compatibility
        features = extract_features("Test", iso_matcher, language_hint='en')
        assert features.shape == (24,)
    
    def test_organization_name(self, iso_matcher):
        features = extract_features("Acme Corp Ltd", iso_matcher)
        assert features.shape == (24,)
        # Should have legal form
        assert features[0] == 1.0  # has_legal_form
    
    def test_person_name(self, iso_matcher):
        features = extract_features("Dr. John Smith Jr.", iso_matcher)
        assert features.shape == (24,)
        # Casing features start at index 16 (8 ISO + 8 struct)
        # has_person_title is index 19 (16 + 3 for all_caps_ratio, capitalized_ratio, initials)
        assert features[19] == 1.0  # has_person_title
        # Lexical features start at index 21 (8+8+5)
        # has_given_name_hit is index 22
        assert features[22] == 1.0  # has_given_name_hit
    
    def test_feature_names_match_count(self):
        names = get_feature_names()
        assert len(names) == 24  # Updated from 32 to 24
    
    def test_feature_names_unique(self):
        names = get_feature_names()
        assert len(names) == len(set(names))
    
    def test_empty_string(self, iso_matcher):
        features = extract_features("", iso_matcher)
        assert features.shape == (24,)
    
    def test_unicode_name(self, iso_matcher):
        features = extract_features("Café François S.A.", iso_matcher)
        assert features.shape == (24,)
    
    def test_all_features_numeric(self, iso_matcher):
        features = extract_features("Test Company Ltd", iso_matcher, language_hint='en')
        assert np.all(np.isfinite(features))
        assert not np.any(np.isnan(features))


class TestGetFeatureNames:
    """Tests for feature name retrieval."""
    
    def test_returns_list(self):
        names = get_feature_names()
        assert isinstance(names, list)
    
    def test_correct_count(self):
        names = get_feature_names()
        assert len(names) == 24  # Updated from 32 to 24
    
    def test_iso_features_present(self):
        names = get_feature_names()
        assert 'has_legal_form' in names
        assert 'legal_form_tier_a' in names
        assert 'legal_form_tier_b' in names
        assert 'legal_form_tier_c' in names
    
    # NOTE: test_language_features_present removed - language features no longer in feature set
    
    def test_structure_features_present(self):
        names = get_feature_names()
        assert 'char_len' in names
        assert 'token_count' in names
    
    def test_casing_features_present(self):
        names = get_feature_names()
        assert 'all_caps_ratio' in names
        assert 'has_initials_pattern' in names
        assert 'has_person_title' in names
    
    def test_lexical_features_present(self):
        names = get_feature_names()
        assert 'has_org_keyword' in names
        assert 'has_given_name_hit' in names
        assert 'has_surname_particle' in names
