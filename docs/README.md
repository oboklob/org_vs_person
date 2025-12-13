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

### [CLASS_IMBALANCE.md](./CLASS_IMBALANCE.md) 📈 **Training Data**
Analysis of class distribution in training data and handling strategies.

---

## Quick Navigation

**Want to understand why the engineered features didn't work?**  
→ Read [LESSONS_LEARNED.md](./LESSONS_LEARNED.md) sections 1-3

**Need full technical details for a report/presentation?**  
→ Read [ENGINEERED_FEATURES_POSTMORTEM.md](./ENGINEERED_FEATURES_POSTMORTEM.md)

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
| 2025-12-12 | README.md | This file |

---

## Contributing

When adding new documentation:
1. Update this README with a link and summary
2. Add entry to Document History table
3. Keep summaries concise - link to details
4. Use clear section headers for easy navigation
