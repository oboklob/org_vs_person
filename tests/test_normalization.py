"""Unit tests for normalization module."""
import pytest
from name_classifier.normalization import normalize, strip_diacritics, NormalizedText


class TestStripDiacritics:
    """Tests for diacritics stripping function."""
    
    def test_removes_common_diacritics(self):
        assert strip_diacritics("café") == "cafe"
        assert strip_diacritics("naïve") == "naive"
        assert strip_diacritics("résumé") == "resume"
    
    def test_removes_various_accents(self):
        assert strip_diacritics("Zürich") == "Zurich"
        assert strip_diacritics("São Paulo") == "Sao Paulo"
        # Note: Ł is a special case that doesn't decompose with NFD
        assert strip_diacritics("Łódź") in ["Łodz", "Lodz"]  # Either is acceptable
    
    def test_preserves_ascii(self):
        assert strip_diacritics("hello world") == "hello world"
        assert strip_diacritics("ABC123") == "ABC123"
    
    def test_empty_string(self):
        assert strip_diacritics("") == ""


class TestNormalize:
    """Tests for text normalization function."""
    
    def test_basic_normalization(self):
        result = normalize("Hello World")
        assert result.raw_str == "Hello World"
        assert result.norm_str == "hello world"
        assert result.tokens == ["hello", "world"]
    
    def test_casefold(self):
        result = normalize("ACME CORPORATION")
        assert result.norm_str == "acme corporation"
        assert result.tokens == ["acme", "corporation"]
    
    def test_punctuation_replacement(self):
        result = normalize("Café & Co., Ltd.")
        assert result.norm_str == "café co ltd"
        assert result.tokens == ["café", "co", "ltd"]
    
    def test_removes_special_characters(self):
        result = normalize("Smith-Jones & Associates, Inc.")
        assert result.norm_str == "smith jones associates inc"
        assert result.tokens == ["smith", "jones", "associates", "inc"]
    
    def test_multiple_spaces_collapsed(self):
        result = normalize("Bob    Smith")
        assert result.norm_str == "bob smith"
        assert result.tokens == ["bob", "smith"]
    
    def test_leading_trailing_whitespace_trimmed(self):
        result = normalize("  Microsoft Corporation  ")
        assert result.norm_str == "microsoft corporation"
        assert result.tokens == ["microsoft", "corporation"]
    
    def test_with_diacritics_removal(self):
        result = normalize("Société Générale", strip_diacritics_flag=True)
        assert result.norm_str == "societe generale"
        assert result.tokens == ["societe", "generale"]
    
    def test_without_diacritics_removal(self):
        result = normalize("Société Générale", strip_diacritics_flag=False)
        assert result.norm_str == "société générale"
        assert result.tokens == ["société", "générale"]
    
    def test_preserves_digits(self):
        result = normalize("Company123")
        assert result.norm_str == "company123"
        assert result.tokens == ["company123"]
    
    def test_ampersand_handling(self):
        result = normalize("Johnson & Johnson")
        assert result.norm_str == "johnson johnson"
        assert result.tokens == ["johnson", "johnson"]
    
    def test_comma_handling(self):
        result = normalize("Smith, John")
        assert result.norm_str == "smith john"
        assert result.tokens == ["smith", "john"]
    
    def test_slash_hyphen_handling(self):
        result = normalize("ABC/DEF-GHI")
        assert result.norm_str == "abc def ghi"
        assert result.tokens == ["abc", "def", "ghi"]
    
    def test_empty_string(self):
        result = normalize("")
        assert result.raw_str == ""
        assert result.norm_str == ""
        assert result.tokens == []
    
    def test_none_input(self):
        result = normalize(None)
        assert result.raw_str == ""
        assert result.norm_str == ""
        assert result.tokens == []
    
    def test_only_punctuation(self):
        result = normalize("!!!")
        assert result.raw_str == "!!!"
        assert result.norm_str == ""
        assert result.tokens == []
    
    def test_mixed_case_with_numbers(self):
        result = normalize("A1B2C3")
        assert result.norm_str == "a1b2c3"
        assert result.tokens == ["a1b2c3"]
    
    def test_initials_pattern(self):
        result = normalize("A. B. Smith")
        assert result.norm_str == "a b smith"
        assert result.tokens == ["a", "b", "smith"]
    
    def test_preserves_raw_string(self):
        original = "Café & Co., Ltd."
        result = normalize(original)
        assert result.raw_str == original
        assert result.raw_str != result.norm_str
    
    def test_legal_form_abbreviation(self):
        result = normalize("Company S.A.")
        assert result.norm_str == "company s a"
        assert result.tokens == ["company", "s", "a"]
    
    def test_unicode_preservation(self):
        result = normalize("北京公司")
        assert result.norm_str == "北京公司"
        assert result.tokens == ["北京公司"]
    
    def test_parentheses_removal(self):
        result = normalize("Company (USA)")
        assert result.norm_str == "company usa"
        assert result.tokens == ["company", "usa"]
    
    def test_quotes_removal(self):
        result = normalize('"The Company"')
        assert result.norm_str == "the company"
        assert result.tokens == ["the", "company"]


class TestNormalizedText:
    """Tests for NormalizedText dataclass."""
    
    def test_dataclass_attributes(self):
        nt = NormalizedText(
            raw_str="Test Inc.",
            norm_str="test inc",
            tokens=["test", "inc"]
        )
        assert nt.raw_str == "Test Inc."
        assert nt.norm_str == "test inc"
        assert nt.tokens == ["test", "inc"]
    
    def test_dataclass_equality(self):
        nt1 = NormalizedText("Test", "test", ["test"])
        nt2 = NormalizedText("Test", "test", ["test"])
        assert nt1 == nt2
    
    def test_dataclass_inequality(self):
        nt1 = NormalizedText("Test1", "test1", ["test1"])
        nt2 = NormalizedText("Test2", "test2", ["test2"])
        assert nt1 != nt2
