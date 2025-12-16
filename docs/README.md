# Name Classifier Documentation

This directory contains documentation for the Name Classifier project.

## Documents

### [LESSONS_LEARNED.md](./LESSONS_LEARNED.md) ⭐ **Start Here**
Quick-reference summary of the engineered features experiment and key takeaways. Read this first for the TL;DR.

**Key Points**:
- Why character n-grams are hard to beat
- When engineered features help (and when they don't)
- Practical recommendations for this project

### [ENGINEERED_FEATURES_POSTMORTEM.md](./ENGINEERED_FEATURES_POSTMORTEM.md) 📊 **Deep Dive**
Comprehensive post-mortem of the engineered features experiment (December 2025).

**Contents**:
- Complete implementation details
- All experiments conducted  
- Critical discoveries and debugging process
- Feature coefficient analysis
- Full code artifact inventory
- Alternative approaches to try

**Stats**: 2,650 lines of code, 129 tests, 0.888 accuracy (worse than 0.91+ baseline)

### [FAST_ORG_DETECTION.md](./FAST_ORG_DETECTION.md) 🚀 **Fast Filter**
Documentation for FastOrgDetector - high-precision organization detection using legal forms.

**Performance**: 99.35-99.97% precision for ORG detection (lower recall)

### [FILTER_ACCURACY_ANALYSIS.md](./FILTER_ACCURACY_ANALYSIS.md) 📊 **Filter Accuracy**
Detailed accuracy analysis of the ISO20275 filter vs ML classifier.

**Key Finding**: Filter achieves 99%+ precision but only 28-54% recall vs classifier's 96% precision and 92% recall.

### [FILTER_PERFORMANCE_ANALYSIS.md](./FILTER_PERFORMANCE_ANALYSIS.md) ⚡ **Filter Performance**
Performance benchmark showing filter is slower than ML for batch processing.

**Recommendation**: Use ML classifier for speed, filter for high-precision use cases.

### [CLASS_IMBALANCE.md](./CLASS_IMBALANCE.md) 📈 **Training Data**
Analysis of class distribution in training data and handling strategies.

### [PROJECT_CLEANUP.md](./PROJECT_CLEANUP.md) 🧹 **Historical**
Record of cleanup performed after engineered features experiment (December 2025).

### [TRAINING_SCRIPT_CLEANUP.md](./TRAINING_SCRIPT_CLEANUP.md) 🔧 **Historical**
Documents changes to `train_model.py` (RandomForest removal, HashingVectorizer addition).

---

## Quick Navigation

**Want to understand why the engineered features didn't work?**  
→ Read [LESSONS_LEARNED.md](./LESSONS_LEARNED.md) sections 1-3

**Need full technical details for a report/presentation?**  
→ Read [ENGINEERED_FEATURES_POSTMORTEM.md](./ENGINEERED_FEATURES_POSTMORTEM.md)

**Want high-precision organization detection?**  
→ Read [FAST_ORG_DETECTION.md](./FAST_ORG_DETECTION.md) and [FILTER_ACCURACY_ANALYSIS.md](./FILTER_ACCURACY_ANALYSIS.md)

**Want to understand filter performance trade-offs?**  
→ Read [FILTER_PERFORMANCE_ANALYSIS.md](./FILTER_PERFORMANCE_ANALYSIS.md)

**Want to improve the classifier?**  
→ Read [LESSONS_LEARNED.md](./LESSONS_LEARNED.md) "Recommendations Going Forward"

**Curious about training data balance?**  
→ Read [CLASS_IMBALANCE.md](./CLASS_IMBALANCE.md)

---

## Key Lessons from This Project

1. **Character n-grams are remarkably effective** for name classification
2. **Manual feature engineering is often redundant** when n-grams work
3. **Validate assumptions early** before building complex systems
4. **More code ≠ better results** - stick with simple baselines
5. **Test coverage ≠ quality** - need end-to-end performance validation

---

## Recommended Approach

Based on our experiments, the best approach for name classification is:

```bash
python scripts/train_model.py \
    --fix-model LogisticRegression \
    --fix-vectorizer TfidfVectorizer \
    --data-dir data
```

**Achieves**: 0.91+ accuracy with minimal complexity

See [LESSONS_LEARNED.md](./LESSONS_LEARNED.md) for alternatives if you need to improve further.

---

## Document History

| Date | Document | Description |
|------|----------|-------------|
| 2025-12-11 | CLASS_IMBALANCE.md | Training data analysis |
| 2025-12-12 | ENGINEERED_FEATURES_POSTMORTEM.md | Full experiment documentation |
| 2025-12-12 | LESSONS_LEARNED.md | Quick-reference summary |
| 2025-12-12 | PROJECT_CLEANUP.md | Cleanup record (historical) |
| 2025-12-12 | TRAINING_SCRIPT_CLEANUP.md | Training script changes (historical) |
| 2025-12-12 | FAST_ORG_DETECTION.md | Fast filter documentation |
| 2025-12-15 | FILTER_ACCURACY_ANALYSIS.md | Filter accuracy benchmarks |
| 2025-12-15 | FILTER_PERFORMANCE_ANALYSIS.md | Filter performance benchmarks |
| 2025-12-15 | README.md | This file (updated) |

---

## Contributing

When adding new documentation:
1. Update this README with a link and summary
2. Add entry to Document History table
3. Keep summaries concise - link to details
4. Use clear section headers for easy navigation
