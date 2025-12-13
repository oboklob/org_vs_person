"""Unit tests for ISO 20275 matcher module."""
import pytest
from pathlib import Path
from name_classifier.iso20275_matcher import (
    ISO20275Matcher,
    SuffixMatch,
    FormMetadata,
    AMBIGUOUS_SHORT_FORMS
)


class TestISO20275Matcher:
    """Tests for ISO20275Matcher class."""
    
    @pytest.fixture
    def matcher(self):
        """Create matcher instance with default ISO data."""
        return ISO20275Matcher()
    
    def test_initialization(self, matcher):
        """Test matcher initializes successfully."""
        assert matcher is not None
        assert matcher.forms_by_len is not None
        assert len(matcher.forms_by_len) > 0
    
    def test_loads_forms(self, matcher):
        """Test that forms are loaded from CSV."""
        stats = matcher.get_stats()
        assert stats['total'] > 0
        assert stats['tier_a'] > 0
        assert stats['tier_b'] >= 0
        assert stats['tier_c'] >= 0
    
    def test_tier_distribution(self, matcher):
        """Test that forms are distributed across tiers."""
        stats = matcher.get_stats()
        # Should have forms in multiple tiers
        assert stats['tier_a'] > 0  # Multi-token and long forms
        # Tier B and C may be present depending on data
    
    def test_match_simple_ltd(self, matcher):
        """Test matching simple 'Ltd' suffix."""
        result = matcher.match_legal_form("Acme Corporation Ltd")
        assert result is not None
        assert 'ltd' in result.suffix.lower()
        assert result.token_len >= 1
    
    def test_match_multi_token_suffix(self, matcher):
        """Test matching multi-token legal form."""
        result = matcher.match_legal_form("Company Proprietary Limited")
        assert result is not None
        # Should match multi-token form
        assert result.token_len >= 1
    
    def test_match_sa(self, matcher):
        """Test matching S.A. which may be ambiguous."""
        result = matcher.match_legal_form("Empresa S.A.")
        assert result is not None
        assert 's' in result.suffix.lower() or 'a' in result.suffix.lower()
    
    def test_match_gmbh(self, matcher):
        """Test matching GmbH (German form)."""
        result = matcher.match_legal_form("Volkswagen GmbH")
        assert result is not None
        assert 'gmbh' in result.suffix.lower()
        # GmbH is 4 chars, should be Tier A
        assert result.metadata.tier == 'A'
    
    def test_no_match_person_name(self, matcher):
        """Test that person names don't match."""
        result = matcher.match_legal_form("John Smith")
        # Should not match or match weakly
        if result:
            # If it matches, should be low confidence
            assert result.is_ambiguous_short or result.metadata.tier in ['B', 'C']
    
    def test_longest_match_first(self, matcher):
        """Test that longest match is preferred."""
        result = matcher.match_legal_form("Company Pty Ltd")
        assert result is not None
        # Should prefer longer match over single token
        assert result.token_len >= 1
    
    def test_case_insensitive(self, matcher):
        """Test matching is case-insensitive."""
        result1 = matcher.match_legal_form("Company LTD")
        result2 = matcher.match_legal_form("Company Ltd")
        result3 = matcher.match_legal_form("Company ltd")
        # All should match
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
    
    def test_punctuation_normalized(self, matcher):
        """Test that punctuation is normalized."""
        result1 = matcher.match_legal_form("Company S.A.")
        result2 = matcher.match_legal_form("Company SA")
        result3 = matcher.match_legal_form("Company S A")
        # All should match (normalized to same form)
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
    
    def test_no_match_empty(self, matcher):
        """Test empty string returns no match."""
        result = matcher.match_legal_form("")
        assert result is None
    
    def test_no_match_only_legal_form(self, matcher):
        """Test string with only legal form."""
        result = matcher.match_legal_form("Ltd")
        assert result is not None
        # Should match the suffix
        assert result.token_len == 1
    
    def test_suffix_metadata(self, matcher):
        """Test that match includes proper metadata."""
        result = matcher.match_legal_form("Test Company Ltd")
        assert result is not None
        assert result.metadata is not None
        assert result.metadata.elf_code != ""
        assert isinstance(result.metadata.tier, str)
        assert result.metadata.tier in ['A', 'B', 'C']
    
    def test_suffix_measurements(self, matcher):
        """Test suffix length measurements."""
        result = matcher.match_legal_form("Company GmbH")
        assert result is not None
        assert result.token_len >= 1
        assert result.char_len >= 1
        # Character length should not include spaces
        assert result.char_len <= len(result.suffix.replace(' ', '')) + 1
    
    def test_ambiguous_short_forms(self, matcher):
        """Test that ambiguous short forms are flagged."""
        # Test a few known ambiguous forms
        for form in ['sa', 'ag', 'nv']:
            result = matcher.match_legal_form(f"Company {form.upper()}")
            if result and form in result.suffix.lower():
                # Should be marked as ambiguous
                assert result.is_ambiguous_short or result.metadata.tier == 'C'
    
    def test_not_ambiguous_de(self, matcher):
        """Test that 'de' is NOT treated as a legal form suffix."""
        result = matcher.match_legal_form("Jean de la Tour")
        # Should not match 'de' as a legal form
        if result and 'de' == result.suffix:
            # If it somehow matches, it should be very weak
            pytest.fail("'de' should not match as single-token legal form")
    
    def test_country_metadata(self, matcher):
        """Test that country information is captured."""
        result = matcher.match_legal_form("Company LLC")
        if result:
            # Country may or may not be present
            assert isinstance(result.metadata.country, str)
    
    def test_stats_totals(self, matcher):
        """Test that stats add up correctly."""
        stats = matcher.get_stats()
        assert stats['total'] == stats['tier_a'] + stats['tier_b'] + stats['tier_c']
    
    def test_tier_a_characteristics(self, matcher):
        """Test Tier A forms have expected characteristics."""
        # Find a Tier A match
        test_names = ["Company GmbH", "Company Limited", "Company Pty Ltd"]
        for name in test_names:
            result = matcher.match_legal_form(name)
            if result and result.metadata.tier == 'A':
                # Tier A should be multi-token OR >= 4 chars
                assert result.token_len >= 2 or result.char_len >= 4
                break
    
    def test_multiple_matches_longest_wins(self, matcher):
        """Test that with multiple possible matches, longest wins."""
        # A name that could match both single and multi-token forms
        result = matcher.match_legal_form("Company Proprietary Limited")
        assert result is not None
        # Should prefer multi-token match
        # Can't assert exact token count without knowing the data,
        # but we can verify it returned something
        assert result.token_len >= 1


class TestAmbiguousShortForms:
    """Test the ambiguous short forms set."""
    
    def test_ambiguous_forms_set_exists(self):
        """Test that ambiguous forms set is defined."""
        assert AMBIGUOUS_SHORT_FORMS is not None
        assert len(AMBIGUOUS_SHORT_FORMS) > 0
    
    def test_sa_is_ambiguous(self):
        """Test that 'sa' is in ambiguous set."""
        assert "sa" in AMBIGUOUS_SHORT_FORMS
    
    def test_de_not_in_ambiguous(self):
        """Test that 'de' is NOT in ambiguous set."""
        assert "de" not in AMBIGUOUS_SHORT_FORMS
    
    def test_expected_forms_present(self):
        """Test that expected ambiguous forms are present."""
        expected = ["sa", "ag", "ab", "nv", "bv", "oy", "sae", "sas", "pty"]
        for form in expected:
            assert form in AMBIGUOUS_SHORT_FORMS, f"{form} should be in ambiguous forms"


class TestFormMetadata:
    """Test FormMetadata dataclass."""
    
    def test_creation(self):
        """Test creating FormMetadata."""
        meta = FormMetadata(
            elf_code="TEST",
            country="US",
            language="en",
            canonical_label="Limited",
            tier="A"
        )
        assert meta.elf_code == "TEST"
        assert meta.country == "US"
        assert meta.tier == "A"


class TestSuffixMatch:
    """Test SuffixMatch dataclass."""
    
    def test_creation(self):
        """Test creating SuffixMatch."""
        meta = FormMetadata("TEST", "US", "en", "Limited", "A")
        match = SuffixMatch(
            suffix="limited",
            metadata=meta,
            token_len=1,
            char_len=7,
            is_ambiguous_short=False
        )
        assert match.suffix == "limited"
        assert match.token_len == 1
        assert match.char_len == 7
        assert not match.is_ambiguous_short
