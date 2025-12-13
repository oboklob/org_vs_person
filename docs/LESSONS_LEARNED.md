# Lessons Learned: Why Simple N-grams Beat Engineered Features

## TL;DR

We spent significant effort adding engineered features (ISO 20275 legal forms, string patterns, lexical signals) to improve name classification. **Result**: Accuracy dropped from 0.91+ to 0.888.

**Why**: Character n-grams already capture everything engineered features provide. Adding explicit features was redundant.

**Recommendation**: Stick with simple TfidfVectorizer + LogisticRegression baseline.

---

## What We Learned

### 1. Character N-grams Are Remarkably Powerful

For the name classification task, character tri-grams and 5-grams capture:
- Legal entity suffixes (ltd, gmbh, inc, corp)
- Common organization patterns (ministry, university, foundation)
- Person name patterns (son, berg, ski, -ov, -ez)
- Language-specific character sequences

**No manual feature engineering needed** - the model discovers these patterns automatically.

### 2. Feature Redundancy Hurts

When we added explicit ISO 20275 legal form matching:
- ✅ Feature extraction worked correctly (24% match rate)
- ✅ Features were in the training pipeline
- ❌ **Model learned 0.0 coefficients** for all ISO features

**Why**: N-grams like "...ltd" and "...gmbh" already perfectly predict organizations. The boolean flag `has_legal_form` provides no additional information.

### 3. More Code ≠ Better Results

We added:
- 2,650 lines of code
- 129 passing tests
- 7 new modules
- Complex feature pipeline

**Result**: Performance got **worse**.

**Lesson**: Validate improvement potential before building. A quick prototype would have revealed this.

### 4. Feature Dropout Can Backfire

We tried 30%/50% dropout on ISO features to prevent over-reliance.

**Result**: Model learned **backwards** - `has_org_keyword` predicted PERSON instead of ORG!

**Why**: Dropout trained model to avoid the features instead of using them properly.

**Lesson**: Don't use dropout on sparse but valuable features.

### 5. Test Coverage ≠ Quality

We achieved 100% test pass rate with comprehensive unit and integration tests.

**But**: Tests verify code works as designed, not whether the design is good.

**Lesson**: Need end-to-end performance validation, not just unit tests.

---

## What Worked Well

### Diagnostic API

The `classify_with_diagnostics()` method is genuinely useful:

```python
result = classifier.classify_with_diagnostics("Acme Corp Inc")
# ClassificationResult(
#    label='ORG', 
#    p_org=0.96,
#    reason_codes={'matched_legal_form': 'inc', 'top_features': [...]}
# )
```

**Recommendation**: Keep this but simplify implementation (remove dependency on engineered features).

### ISO 20275 Data

The legal entity form database is valuable reference data:
- 3,493 forms across 217 jurisdictions
- Tier classification by ambiguity
- Useful for entity extraction tasks

**Recommendation**: Keep as reference documentation.

---

## When Would Engineered Features Help?

Engineered features make sense when:

1. **N-grams can't capture the pattern**
   - Example: Numeric ratios, external knowledge, URL patterns
   
2. **Features have high information density**
   - Example: "Is this a valid email?" (yes/no is more informative than character patterns)

3. **Model needs interpretability**
   - Example: Regulatory compliance requiring explainable features

4. **Sparse high-value signals**
   - Example: Verified business registry match

For name classification, **none of these apply**. N-grams are sufficient.

---

## Recommendations Going Forward

### Keep It Simple
```bash
# This works best:
python scripts/train_model.py \
    --fix-model LogisticRegression \
    --fix-vectorizer TfidfVectorizer \
    --data-dir data
```

Achieves **0.91+ accuracy** with minimal complexity.

### If You Want to Improve Further

Try these instead of engineered features:

1. **Bigger N-gram ranges**: Try (2,6) or (3,7)
2. **Ensemble**: Average LogisticRegression + SGD predictions
3. **More training data**: Data quality > feature quality
4. **Domain-specific rules**: Post-processing for known edge cases

### Archive, Don't Delete

The engineered features code is:
- ✅ Well-implemented
- ✅ Fully tested
- ✅ Educational

**Recommendation**: Keep in separate branch `feature/engineered-features` as reference.

---

## Reading

For full details, see: [`ENGINEERED_FEATURES_POSTMORTEM.md`](./ENGINEERED_FEATURES_POSTMORTEM.md)

---

## Bottom Line

> "Premature optimization is the root of all evil." - Donald Knuth

We optimized before validating the approach would work. A 100-line prototype with 100 examples would have shown this wouldn't help.

**Lesson**: Validate assumptions early. Simple baselines are hard to beat.
