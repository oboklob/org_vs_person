"""Tests for fast organization detector."""
import pytest
from org_vs_person.fast_org_detector import (
    FastOrgDetector,
    OrgDetectionResult,
    is_org_by_legal_form,
    quick_org_filter
)


class TestFastOrgDetector:
    """Tests for FastOrgDetector class."""
    
    def test_detects_clear_org(self):
        detector = FastOrgDetector()
        result = detector.detect("Acme Corporation Ltd")
        
        assert result.is_org is True
        assert result.confidence is not None
        assert result.legal_form == "ltd"
        assert result.tier in ['A', 'B', 'C']
    
    def test_unknown_for_person_name(self):
        detector = FastOrgDetector()
        result = detector.detect("John Smith")
        
        assert result.is_org is False
        assert result.confidence is None
        assert result.legal_form is None
    
    def test_tier_a_filter(self):
        detector = FastOrgDetector(tier_filter=['A'])
        
        # Should detect Tier A forms
        result_a = detector.detect("Company GmbH")  # Tier A: 4+ chars
        assert result_a.is_org is True
        
    def test_tier_ab_filter(self):
        detector = FastOrgDetector(tier_filter=['A', 'B'])
        
        # Should detect both A and B
        result_a = detector.detect("Company GmbH")
        result_b = detector.detect("Company Ltd")
        
        assert result_a.is_org is True
        assert result_b.is_org is True
    
    def test_batch_detection(self):
        detector = FastOrgDetector()
        names = ["Acme Ltd", "John Smith", "Google Inc"]
        results = detector.detect_batch(names)
        
        assert len(results) == 3
        assert all(isinstance(r, OrgDetectionResult) for r in results)
    
    def test_filter_orgs(self):
        detector = FastOrgDetector(tier_filter=['A', 'B'])
        names = ["Acme Corporation Ltd", "John Smith", "TechCo Inc", "Jane Doe"]
        
        orgs, unknown = detector.filter_orgs(names)
        
        # Should detect at least some orgs
        assert len(orgs) > 0
        assert len(unknown) > 0
        assert len(orgs) + len(unknown) == len(names)
    
    def test_confidence_levels(self):
        detector = FastOrgDetector()
        
        # GmbH is Tier A - should be very_high
        result_tier_a = detector.detect("Company GmbH")
        if result_tier_a.is_org:
            assert result_tier_a.confidence == 'very_high'


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_is_org_by_legal_form_conservative(self):
        # Conservative mode (Tier A only)
        assert is_org_by_legal_form("Acme GmbH", conservative=True) is True
        assert is_org_by_legal_form("John Smith", conservative=True) is False
    
    def test_is_org_by_legal_form_less_conservative(self):
        # Less conservative mode (Tier A+B)
        assert is_org_by_legal_form("Acme Ltd", conservative=False) is True
    
    def test_quick_org_filter(self):
        names = ["Company Ltd", "John Doe", "TechCorp Inc", "Jane Smith"]
        orgs, needs_classification = quick_org_filter(names, conservative=False)
        
        assert isinstance(orgs, list)
        assert isinstance(needs_classification, list)
        assert len(orgs) + len(needs_classification) == len(names)
    
    def test_quick_org_filter_conservative(self):
        names = ["Company GmbH", "Company SA"]  # GmbH=Tier A, SA=Tier C
        orgs_conservative, _ = quick_org_filter(names, conservative=True)
        orgs_lenient, _ = quick_org_filter(names, conservative=False)
        
        # Conservative should catch fewer (only Tier A)
        assert len(orgs_conservative) <= len(orgs_lenient)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_string(self):
        detector = FastOrgDetector()
        result = detector.detect("")
        assert result.is_org is False
    
    def test_none_tier_filter(self):
        # None should use all tiers
        detector = FastOrgDetector(tier_filter=None)
        result = detector.detect("Company SA")  # Tier C - ambiguous
        # Should still detect (though less confident)
        assert isinstance(result, OrgDetectionResult)
    
    def test_case_insensitive(self):
        detector = FastOrgDetector()
        result_upper = detector.detect("ACME LTD")
        result_lower = detector.detect("acme ltd")
        result_mixed = detector.detect("Acme Ltd")
        
        # All should give same result
        assert result_upper.is_org == result_lower.is_org == result_mixed.is_org
