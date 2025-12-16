# Archive: Failed Feature Engineering Experiment

This directory contains code from the engineered features experiment (December 2025) that did not improve model performance.

## What's Here

### Production Code (~1,900 lines)
- `train_model_enhanced.py` - Enhanced training script with feature engineering
- `feature_engineering.py` - 32→24 engineered features extraction
- `transformers.py` - Sklearn transformers for features
- `feature_dropout.py` - Tier-conditional dropout (backfired)

### Diagnostic Scripts
- `diagnose_performance.py` - Performance diagnostics
- `analyze_features.py` - Feature importance analysis
- `debug_features.py` - Feature extraction debugging
- `debug_iso_features.py` - ISO matching verification
- `debug_pipeline_order.py` - Pipeline ordering checks
- `check_model_classes.py` - Model class ordering verification

### Tests (~750 lines)
- `test_feature_engineering.py` - Feature extraction tests (45 passing)
- `test_training_integration.py` - Pipeline integration tests
- `test_classifier_diagnostics.py` - Diagnostic API tests

## Why Archived

**Performance**: Accuracy dropped from 0.91+ (baseline) to 0.888 (enhanced)

**Root Cause**: Character n-grams already capture all information that engineered features provide. Explicit features were redundant.

**Key Finding**: Model learned 0.0 coefficients for all 8 ISO legal form features - they provided no additional information beyond n-grams.

## Documentation

See comprehensive analysis in:
- `../docs/ENGINEERED_FEATURES_POSTMORTEM.md` - Full technical deep dive
- `../docs/LESSONS_LEARNED.md` - Quick reference summary

## What Was Kept

The following components **were** useful and remain in production:
- `fast_org_detector.py` - Rule-based org detection (99.58% precision)
- `iso20275_matcher.py` - ISO 20275 legal form matching
- `normalization.py` - Text normalization utilities

## Status

All code is:
- ✅ Functional and tested (100% pass rate when archived)
- ✅ Well-documented with comprehensive docstrings
- ✅ Just not beneficial for model performance

This archive serves as reference for:
- Understanding what was tried and why it didn't work
- Educational material on feature engineering pitfalls
- Potential reuse in different problem domains

---

**Archived**: December 12, 2025  
**Experiment Duration**: ~1 week  
**Total Code**: ~2,650 lines  
**Result**: Documented successful failure
