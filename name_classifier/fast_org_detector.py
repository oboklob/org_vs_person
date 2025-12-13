"""Fast organization detection using legal form suffixes.

This module provides a high-precision rule-based detector that can quickly
identify organizations by checking for legal entity form suffixes (e.g., "Ltd",
"GmbH", "Inc"). It's designed as a fast pre-filter before ML classification.

Performance on 1M test set:
- Tier A only: 99.97% precision, 28.6% recall (14% coverage)
- Tier A+B: 99.58% precision, 52.4% recall (26% coverage)  
- All tiers: 99.41% precision, 54.3% recall (27% coverage)
"""
from typing import Optional, Literal
from dataclasses import dataclass

from name_classifier.iso20275_matcher import ISO20275Matcher, SuffixMatch


@dataclass
class OrgDetectionResult:
    """Result from organization detection.
    
    Attributes:
        is_org: True if detected as organization, False if unknown
        confidence: Confidence level ('very_high', 'high', 'medium', or None)
        legal_form: Matched legal form suffix or None
        tier: Legal form tier ('A', 'B', 'C') or None
    """
    is_org: bool
    confidence: Optional[Literal['very_high', 'high', 'medium']]
    legal_form: Optional[str]
    tier: Optional[str]


class FastOrgDetector:
    """Fast organization detector using legal form suffixes.
    
    This detector achieves >99% precision when detecting organizations by
    checking for legal entity form suffixes. It can be used as a fast pre-filter
    before passing names to an ML classifier.
    
    Example:
        >>> detector = FastOrgDetector(tier_filter=['A'])
        >>> result = detector.detect("Acme Corporation Ltd")
        >>> if result.is_org:
        ...     print(f"Detected ORG with {result.confidence} confidence")
        
    Performance trade-offs:
        - tier_filter=['A']: 99.97% precision, 14% coverage (very conservative)
        - tier_filter=['A', 'B']: 99.58% precision, 26% coverage (recommended)
        - tier_filter=None: 99.41% precision, 27% coverage (most coverage)
    """
    
    def __init__(self, tier_filter: Optional[list] = None):
        """Initialize organization detector.
        
        Args:
            tier_filter: Which confidence tiers to use for detection.
                None = all tiers (A, B, C)
                ['A'] = only highest confidence (99.97% precision)
                ['A', 'B'] = exclude ambiguous (99.58% precision, recommended)
        """
        self.iso_matcher = ISO20275Matcher()
        self.tier_filter = tier_filter
        
    def detect(self, name: str) -> OrgDetectionResult:
        """Detect if name is an organization based on legal form suffix.
        
        Args:
            name: Name to check
            
        Returns:
            OrgDetectionResult with detection outcome and metadata
        """
        match = self.iso_matcher.match_legal_form(name)
        
        if not match:
            # No legal form found - unknown
            return OrgDetectionResult(
                is_org=False,
                confidence=None,
                legal_form=None,
                tier=None
            )
        
        # Check tier filter
        if self.tier_filter and match.metadata.tier not in self.tier_filter:
            # Legal form found but tier filtered out - unknown
            return OrgDetectionResult(
                is_org=False,
                confidence=None,
                legal_form=match.suffix,
                tier=match.metadata.tier
            )
        
        # Detected as organization!
        confidence = self._get_confidence(match)
        
        return OrgDetectionResult(
            is_org=True,
            confidence=confidence,
            legal_form=match.suffix,
            tier=match.metadata.tier
        )
    
    def _get_confidence(self, match: SuffixMatch) -> Literal['very_high', 'high', 'medium']:
        """Determine confidence level based on match characteristics.
        
        Args:
            match: SuffixMatch from ISO matcher
            
        Returns:
            Confidence level string
        """
        # Tier A forms are very high confidence (>99.9% precision)
        if match.metadata.tier == 'A':
            return 'very_high'
        
        # Tier B forms are high confidence (~99% precision)
        if match.metadata.tier == 'B' and not match.is_ambiguous_short:
            return 'high'
        
        # Everything else is medium confidence
        return 'medium'
    
    def detect_batch(self, names: list) -> list[OrgDetectionResult]:
        """Detect organizations in batch.
        
        Args:
            names: List of names to check
            
        Returns:
            List of OrgDetectionResult, one per input name
        """
        return [self.detect(name) for name in names]
    
    def filter_orgs(self, names: list) -> tuple[list, list]:
        """Split names into detected orgs and unknown.
        
        This is useful for pre-filtering: detected orgs can skip ML classification.
        
        Args:
            names: List of names to filter
            
        Returns:
            Tuple of (detected_orgs, unknown_names)
        """
        detected_orgs = []
        unknown = []
        
        for name in names:
            result = self.detect(name)
            if result.is_org:
                detected_orgs.append(name)
            else:
                unknown.append(name)
        
        return detected_orgs, unknown


# Convenience functions for common use cases

def is_org_by_legal_form(name: str, conservative: bool = True) -> bool:
    """Quick check if name is likely an organization based on legal form.
    
    Args:
        name: Name to check
        conservative: If True, use only Tier A (99.97% precision)
                     If False, use Tier A+B (99.58% precision, more coverage)
    
    Returns:
        True if high-confidence organization detection, False otherwise
    """
    detector = FastOrgDetector(tier_filter=['A'] if conservative else ['A', 'B'])
    result = detector.detect(name)
    return result.is_org


def quick_org_filter(names: list, conservative: bool = True) -> tuple[list, list]:
    """Quickly separate likely orgs from names needing ML classification.
    
    Args:
        names: List of names to filter
        conservative: If True, use only Tier A (99.97% precision)
                     If False, use Tier A+B (99.58% precision, more coverage)
    
    Returns:
        Tuple of (detected_orgs, needs_classification)
        
    Example:
        >>> names = ["Acme Ltd", "John Smith", "Google Inc", "Jane Doe"]
        >>> orgs, unknown = quick_org_filter(names)
        >>> print(f"{len(orgs)} quick detections, {len(unknown)} need ML")
    """
    detector = FastOrgDetector(tier_filter=['A'] if conservative else ['A', 'B'])
    return detector.filter_orgs(names)
