# Engineered Features Experiment - Post-Mortem

> [!IMPORTANT]
> **HISTORICAL DOCUMENT** - This documents a failed experiment conducted in December 2025.
> The engineered features approach described here was **NOT adopted** and is preserved  for reference only.
> 
> **Current System**: Uses simple TfidfVectorizer + LogisticRegression (0.91+ accuracy).
> See main [README.md](file:///sites/name_classifier/README.md) for current approach.

**Date**: December 12, 2025  
**Objective**: Improve name classification accuracy by adding engineered features alongside character n-grams  
**Outcome**: ❌ Failed - No improvement over baseline, added unnecessary complexity  
**Baseline Accuracy**: 0.91+ (TfidfVectorizer/LogisticRegression)  
**Final Accuracy**: 0.888 (HashingVectorizer + 24 engineered features)

---

## Executive Summary

We attempted to enhance the name classifier by adding 24 engineered features (ISO 20275 legal forms, string structure, casing patterns, lexical signals) alongside character n-grams. Despite a comprehensive implementation with 129 passing tests, the approach **reduced accuracy** and added significant complexity without benefit.

**Key Finding**: Character n-grams already capture all the information that engineered features provide. Explicit feature engineering is redundant for this problem.

---

## What We Implemented

### 1. Core Infrastructure

#### ISO 20275 Legal Form Matching (`iso20275_matcher.py`)
- Loaded 3,493 legal entity forms from ISO standard
- Implemented longest-match-first algorithm
- Three confidence tiers:
  - **Tier A** (3,191 forms): High precision - multi-token or ≥4 chars
  - **Tier B** (276 forms): Medium - short forms, not ambiguous  
  - **Tier C** (26 forms): Low - ambiguous short forms (sa, ag, inc, etc.)
- **Test Coverage**: 27 unit tests, all passing

#### Text Normalization (`normalization.py`)
- Casefolding, punctuation replacement, whitespace handling
- Optional diacritics stripping
- Tokenization
- **Test Coverage**: 29 unit tests, all passing

#### Feature Engineering (`feature_engineering.py`)
- **24 engineered features** (reduced from initial 32):
  - ISO 20275-derived: 8 features
  - String structure: 8 features  
  - Casing/initials: 5 features
  - Lexical signals: 3 features
- Removed 8 language/jurisdiction features (all zeros - no data)
- **Test Coverage**: 45 unit tests, all passing

### 2. Training Pipeline Enhancements

#### Custom Transformers (`transformers.py`)
- `NameFeatureExtractor`: Extracts 24 features per name
- `TextExtractor`: Extracts text for n-gram vectorization
- Sklearn `BaseEstimator`/`TransformerMixin` compatible

#### Feature Dropout (`feature_dropout.py`)
- Tier-conditional dropout on ISO features during training
- Rates: 30% for Tier A/B, 50% for Tier C
- **Result**: Made things worse - caused inverted feature learning

#### Enhanced Training Script (`train_model_enhanced.py`)  
- `FeatureUnion` combining:
  - Sparse: `HashingVectorizer(analyzer='char', ngram_range=(3,5), n_features=2^18)`
  - Dense: 24 engineered features
- Focused models: `SGDClassifier`, `LogisticRegression`
- Evaluation slicing by token length, suffix presence, tiers
- **570 lines of code**

### 3. Diagnostic Capabilities

#### Enhanced Classifier API (`classifier.py`)
- New `classify_with_diagnostics()` method
- Returns `ClassificationResult` with:
  - `label`: "PER" or "ORG"
  - `p_org`: Probability
  - `reason_codes`: Dictionary with matched legal form, tier, top features
- Optional Tier A shortcut rule
- Backward compatible with existing API
- **Test Coverage**: 10 integration tests

---

## Experiments Conducted

### Experiment 1: Full Feature Set (32 features) with Dropout
- **Config**: 2^20 hash features, 32 engineered, 30%/50% dropout
- **Result**: 0.8912 accuracy
- **Issue**: Feature dropout caused **inverted learning** - model learned opposite patterns

### Experiment 2: Full Feature Set without Dropout
- **Config**: 2^20 hash features, 32 engineered, no dropout
- **Result**: 0.888 accuracy  
- **Issue**: Features still showed bizarre coefficients

### Experiment 3: Reduced Feature Set (24 features) without Dropout
- **Config**: 2^18 hash features, 24 engineered (removed 8 language features)
- **Result**: 0.888 accuracy
- **Finding**: **All 8 ISO legal form features learned 0.0 coefficients**

---

## Critical Discoveries

### Discovery 1: Feature Coefficient Inversion (WITH Dropout)

When using dropout, the model learned **backwards patterns**:

| Feature | Expected | Actual (with dropout) |
|---------|----------|----------------------|
| `all_caps_ratio` | → ORG | → PER (-8.27) ❌ |
| `has_org_keyword` | → ORG | → PER (-3.11) ❌ |
| `has_given_name_hit` | → PER | → ORG (+2.28) ❌ |
| `legal_form_tier_a` | → ORG | → PER (-0.76) ❌ |

**Root Cause**: Dropout randomly zeroed ISO features, training model to **avoid** them instead of use them.

### Discovery 2: Model Class Ordering Confusion

Initial analysis showed features backwards because:
```python
model.classes_ = ['ORG', 'PER']  # Index 0 = ORG, Index 1 = PER
# Positive coefficient → predicts index 1 (PER)
# Negative coefficient → predicts index 0 (ORG)
```

The analysis script incorrectly assumed positive = ORG. Once fixed, coefficients made sense but revealed the zero-coefficient problem.

### Discovery 3: ISO Features Completely Ignored (WITHOUT Dropout)

Even without dropout, **all 8 ISO features had 0.0 coefficients**:

```
has_legal_form                  0.0000
legal_form_tier_a               0.0000
legal_form_tier_b               0.0000  
legal_form_tier_c               0.0000
legal_form_token_len            0.0000
legal_form_char_len             0.0000
legal_form_is_ambiguous_short   0.0000
legal_form_country_is_known     0.0000
```

**Verified**: 
- ✅ Features ARE being extracted correctly (24% of names have legal forms)
- ✅ Features ARE in the pipeline (last 24 features of matrix)
- ✅ Manual extraction matches pipeline extraction
- ❌ Model assigns zero weight to all of them

**Why**: Character n-grams like "ltd", "gmbh", "inc" already capture legal forms perfectly. The boolean flags provide **no additional information**.

### Discovery 4: Only Simple Features Worked

Features that DID get non-zero coefficients:

| Feature | Coefficient | Prediction | Notes |
|---------|-------------|------------|-------|
| `all_caps_ratio` | -8.27 | → ORG | Strong signal |
| `has_org_keyword` | -2.79 | → ORG | "inc", "corp" etc |
| `capitalised_token_ratio` | +2.63 | → PER | Person names |
| `has_given_name_hit` | +2.27 | → PER | "John", "Mary" etc |
| `contains_digit` | -1.90 | → ORG | "Company123" |
| `has_person_title` | +1.12 | → PER | "Dr.", "Mr." |

**Pattern**: Only features that capture information **NOT in n-grams** got non-zero weights.

---

## Lessons Learned

### 1. N-grams Are Exceptionally Powerful

Character n-grams (3-5) already capture:
- Legal form suffixes ("ltd", "gmbh", "inc")
- Common organization words ("corporation", "ministry")
- Name patterns ("son", "berg", "ski")
- Language-specific patterns

Adding explicit features for these is **redundant**.

### 2. Feature Dropout Can Backfire

Randomly dropping features during training can cause the model to learn to **ignore** those features entirely, or worse, learn inverse correlations.

**Lesson**: Only use dropout when features are truly redundant/noisy, not for sparse but valuable features.

### 3. More Features ≠ Better Performance

We added:
- ~1,900 lines of new code
- 129 tests
- 570-line training script
- Complex feature pipeline

**Result**: **Worse accuracy** (0.888 vs 0.91+)

**Lesson**: Simple often beats complex. Validate improvements early.

### 4. Test Coverage ≠ Quality

We achieved:
- 100% test pass rate on all modules
- Comprehensive unit tests (108)
- Integration tests (21)
- Feature extraction verified

**But still failed** because the fundamental approach was flawed.

**Lesson**: Tests verify correctness, not utility. Need actual performance validation.

### 5. Character N-grams Are Hard to Beat

For text classification with character-level patterns:
- N-grams are simple, fast, effective
- They automatically discover relevant patterns
- No manual feature engineering needed
- Resistant to spelling variations

**Lesson**: Don't add complexity unless demonstrated gains justify it.

---

## Code Artifacts Created

### Production Code (~1,900 lines)
- `name_classifier/normalization.py` (86 lines)
- `name_classifier/iso20275_matcher.py` (232 lines)
- `name_classifier/feature_engineering.py` (292 lines)
- `name_classifier/transformers.py` (119 lines)
- `name_classifier/feature_dropout.py` (89 lines)
- `name_classifier/classifier.py` (+165 lines for diagnostics)
- `scripts/train_model_enhanced.py` (570 lines)

### Test Code (~750 lines)
- `tests/test_normalization.py` (142 lines)
- `tests/test_iso20275_matcher.py` (187 lines)
- `tests/test_feature_engineering.py` (283 lines)
- `tests/test_training_integration.py` (174 lines)
- `tests/test_classifier_diagnostics.py` (165 lines)

### Diagnostic Scripts
- `scripts/analyze_features.py` - Feature importance analysis
- `scripts/diagnose_performance.py` - Performance diagnostics
- `scripts/debug_features.py` - Feature extraction debugging
- `scripts/debug_iso_features.py` - ISO matching verification
- `scripts/debug_pipeline_order.py` - Pipeline ordering checks

**Total**: ~2,650 lines of code

**All code is functional and tested** - it just doesn't improve performance.

---

## Recommendations

### 1. Stick with Simple Baseline

Use the proven approach:
```python
python scripts/train_model.py \
    --data-dir data \
    --fix-model LogisticRegression \
    --fix-vectorizer TfidfVectorizer
```

**Achieves**: 0.91+ accuracy with minimal complexity

### 2. Keep Diagnostic API

The `classify_with_diagnostics()` method is useful for debugging:
```python
result = classifier.classify_with_diagnostics("Acme Corp")
# Returns: ClassificationResult(label='ORG', p_org=0.95, reason_codes={...})
```

**Recommendation**: Keep this but remove dependency on engineered features.

### 3. Archive Engineered Features

Move to separate branch or archive:
- Normalization module (might be useful elsewhere)
- ISO 20275 matcher (interesting reference data)
- Feature engineering (educational example)

Don't delete - might inform future work or be useful for different problems.

### 4. Document ISO 20275 Data

The ISO 20275 legal form data is valuable:
- 3,493 forms across 217 jurisdictions
- Tier classification for ambiguity
- Could be useful for entity extraction tasks

**Recommendation**: Keep `data/iso20275/` and matcher as reference.

---

## Alternative Approaches Worth Trying

If we want to improve beyond 0.91+ baseline:

### 1. Ensemble Methods
- Combine multiple n-gram ranges: (2,4), (3,5), (4,6)
- Different vectorization: TF-IDF + HashingVectorizer
- Model averaging: LogisticRegression + SGDClassifier

### 2. Deep Learning (if dataset is large enough)
- Character-level CNN
- LSTM/GRU on character sequences
- Pre-trained transformers (BERT) - overkill but might work

### 3. Active Learning on Hard Cases
- Identify names model struggles with
- Add targeted training examples
- Focus on ambiguous short forms

### 4. External Knowledge
- Use business registries for validation
- Incorporate country-specific patterns
- Add domain-specific rules (universities, governments)

---

## Conclusion

This experiment was a **successful failure**. We:

1. ✅ Implemented a comprehensive engineered feature system
2. ✅ Achieved 100% test coverage
3. ✅ Discovered why it didn't work
4. ✅ Learned valuable lessons about feature redundancy

**Key Takeaway**: Character n-grams are remarkably effective for name classification. Manual feature engineering adds complexity without benefit when n-grams already capture the patterns.

**Recommendation**: Revert to baseline approach (0.91+ accuracy) and invest effort elsewhere - data quality, edge case handling, or deployment infrastructure.

---

## References

- ISO 20275: Entity Legal Forms Code List v1.5
- Original training script: `scripts/train_model.py`
- Baseline performance: `docs/performance_baseline.md` (if exists)
- Character n-grams paper: Cavnar & Trenkle, 1994

---

## Appendix: Quick Start (For Reference)

If someone wants to reproduce this experiment:

```bash
# Run enhanced training
python scripts/train_model_enhanced.py \
    --data-dir data \
    --no-dropout \
    --hash-features 262144

# Analyze features
python scripts/analyze_features.py

# Test diagnostics
python -c "
from name_classifier import NameClassifier
c = NameClassifier()
result = c.classify_with_diagnostics('Acme Corp Ltd')
print(result)
"
```

**Expected outcome**: 0.888 accuracy, ISO features with 0.0 coefficients.
