"""Name classifier package for distinguishing between person and organization names."""
from name_classifier.classifier import NameClassifier, ClassificationResult
from name_classifier.normalization import normalize, NormalizedText
from name_classifier.iso20275_matcher import ISO20275Matcher, SuffixMatch, FormMetadata
from name_classifier.fast_org_detector import (
    FastOrgDetector,
    OrgDetectionResult,
    is_org_by_legal_form,
    quick_org_filter
)

__version__ = "0.1.0"

__all__ = [
    "NameClassifier",
    "ClassificationResult",
    "normalize",
    "NormalizedText",
    "ISO20275Matcher",
    "SuffixMatch",
    "FormMetadata",
    "FastOrgDetector",
    "OrgDetectionResult",
    "is_org_by_legal_form",
    "quick_org_filter",
]
