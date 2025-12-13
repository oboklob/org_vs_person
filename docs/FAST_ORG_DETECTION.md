# Fast Organization Detection

## Overview

The `FastOrgDetector` provides **99%+ precision** organization detection using legal entity form suffixes. It's designed as a fast pre-filter before ML classification.

## Performance (on 1M test set)

| Configuration | Precision | Recall | Coverage | Use Case |
|--------------|-----------|--------|----------|----------|
| **Tier A only** | 99.97% | 28.6% | 14% | Maximum precision |
| **Tier A+B** (recommended) | 99.58% | 52.4% | 26% | Balance precision/coverage |
| **All tiers** | 99.41% | 54.3% | 27% | Maximum coverage |

## Quick Start

### Simple Detection

```python
from name_classifier import is_org_by_legal_form

# Quick check with conservative settings (99.97% precision)
if is_org_by_legal_form("Acme Corporation Ltd"):
    print("High-confidence ORG detection!")

# Less conservative (99.58% precision, more coverage)
if is_org_by_legal_form("Acme Corp SA", conservative=False):
    print("Detected as ORG")
```

### Batch Filtering

```python
from name_classifier import quick_org_filter

names = [
    "Acme Corporation Ltd",
    "John Smith",
    "Google Inc",
    "Jane Doe",
    "Microsoft Corp"
]

# Split into detected orgs and names needing ML classification
detected_orgs, needs_ml = quick_org_filter(names, conservative=False)

print(f"{len(detected_orgs)} orgs detected instantly")
print(f"{len(needs_ml)} names need ML classifier")
# Output: 3 orgs detected instantly, 2 names need ML classifier
```

### Full API

```python
from name_classifier import FastOrgDetector

# Initialize with tier filter
detector = FastOrgDetector(tier_filter=['A', 'B'])  # Recommended

# Detect single name
result = detector.detect("Acme Corporation GmbH")
print(f"Is ORG: {result.is_org}")
print(f"Confidence: {result.confidence}")
print(f"Legal form: {result.legal_form}")
print(f"Tier: {result.tier}")

# Batch detection
results = detector.detect_batch([
    "Company Ltd",
    "John Smith",
    "TechCorp Inc"
])

# Filter for processing pipeline
orgs, unknown = detector.filter_orgs(name_list)
# Process orgs immediately, send unknown to ML classifier
```

## How It Works

1. **Normalization**: Name is case-normalized and tokenized
2. **Suffix Matching**: Checks last 6 tokens for legal form patterns
3. **Tier Classification**: Matches are classified into confidence tiers:
   - **Tier A** (3,191 forms): Multi-token OR ≥4 characters (e.g., "GmbH", "Limited")
   - **Tier B** (276 forms): 2-3 char, non-ambiguous (e.g., "Ltd", "Inc")
   - **Tier C** (26 forms): Ambiguous short forms (e.g., "SA", "AG")

4. **Precision Guarantee**: >99% confidence when detecting ORG

## Configuration Options

### Tier Filters

```python
# Maximum precision (99.97%)
detector = FastOrgDetector(tier_filter=['A'])

# Recommended balance (99.58% precision, 26% coverage)
detector = FastOrgDetector(tier_filter=['A', 'B'])

# Maximum coverage (99.41% precision, 27% coverage)
detector = FastOrgDetector(tier_filter=None)  # or ['A', 'B', 'C']
```

### Integration with ML Classifier

```python
from name_classifier import FastOrgDetector, NameClassifier

# Initialize both
fast_detector = FastOrgDetector(tier_filter=['A', 'B'])
ml_classifier = NameClassifier()

def classify_name(name):
    """Hybrid approach: fast detection + ML fallback."""
    # Try fast detection first
    result = fast_detector.detect(name)
    
    if result.is_org:
        # High-confidence ORG - skip ML
        return {
            'label': 'ORG',
            'method': 'legal_form',
            'confidence': result.confidence,
            'legal_form': result.legal_form
        }
    
    # Unknown - use ML classifier
    ml_label = ml_classifier.classify(name)
    return {
        'label': ml_label,
        'method': 'ml_classifier',
        'confidence': 'medium'
    }

# Example usage
print(classify_name("Acme Corporation Ltd"))
# {'label': 'ORG', 'method': 'legal_form', 'confidence': 'very_high', ...}

print(classify_name("John Smith"))
# {'label': 'PER', 'method': 'ml_classifier', 'confidence': 'medium'}
```

## Performance Benefits

### Speed

- **Legal form matching**: O(tokens) - very fast
- **ML classification**: O(n-grams) - slower
- **Speedup**: ~26% of names skip ML entirely (Tier A+B config)

### Accuracy

With tier='A+B' configuration:
- **26% of names**: Classified with 99.58% precision instantly
- **74% of names**: Sent to ML classifier (0.91+ accuracy)
- **Combined accuracy**: Better than ML alone

### Example: 1M Name Pipeline

```
Total names: 1,000,000
├─ Fast detection (Tier A+B): 258,600 → ORG (99.58% precision)
└─ ML classifier: 741,400 → Various (91%+ accuracy)

Time saved: ~26% reduction in ML calls
Error rate: <0.5% on fast-detected names
```

## API Reference

### FastOrgDetector

```python
class FastOrgDetector(tier_filter=None)
```

**Parameters**:
- `tier_filter`: Which tiers to use (`['A']`, `['A', 'B']`, or `None`)

**Methods**:
- `detect(name)` → `OrgDetectionResult`
- `detect_batch(names)` → `List[OrgDetectionResult]`
- `filter_orgs(names)` → `(detected_orgs, unknown_names)`

### OrgDetectionResult

```python
@dataclass
class OrgDetectionResult:
    is_org: bool  # True if detected as ORG
    confidence: Optional[str]  # 'very_high', 'high', 'medium', or None
    legal_form: Optional[str]  # Matched suffix (e.g., "ltd")
    tier: Optional[str]  # 'A', 'B', 'C', or None
```

### Convenience Functions

```python
is_org_by_legal_form(name, conservative=True) → bool
quick_org_filter(names, conservative=True) → (orgs, unknown)
```

## Limitations

1. **Only detects ORG, not PER**: Returns `is_org=False` for unknown, not definitive PER
2. **Suffix-based only**: Won't detect orgs without legal form suffixes
3. **Language coverage**: Best for Latin script, ~217 jurisdictions
4. **Precision over recall**: Optimized to avoid false positives, will miss some orgs

## Best Practices

### DO

✅ Use as pre-filter before ML classification  
✅ Trust high-confidence detections (Tier A)  
✅ Use Tier A+B for production (99.58% precision)  
✅ Monitor false positive rate in your domain

### DON'T

❌ Rely solely on this for all classification  
❌ Use Tier C alone (too many ambiguous forms)  
❌ Assume unknown = person (could be org without suffix)  
❌ Skip validation on your specific dataset

## Testing

Run the performance test on your data:

```bash
python scripts/test_legal_form_precision.py
```

This will show precision/recall/coverage for all tier configurations on your test set.

## See Also

- [ISO 20275 Standard](https://www.iso.org/standard/67420.html)
- [Legal Entity Forms Data](../data/iso20275/)
- [`ISO20275Matcher` Documentation](../name_classifier/iso20275_matcher.py)
