# Handling Class Imbalance in the Name Classifier

## The Problem

The paranames dataset has a significant class imbalance:
- **PER (Person):** 93M entries (~85%)
- **ORG (Organization):** 16M entries (~15%)

**Ratio:** Approximately **6:1** (PER:ORG)

### Why This Matters

Without handling class imbalance, machine learning models tend to:
1. **Bias toward the majority class** - Predict "PER" more often
2. **Achieve misleadingly high accuracy** - Could get 85% accuracy by always predicting "PER"!
3. **Perform poorly on minority class** - ORG classification suffers

## Our Solutions

### 1. ✅ Stratified Sampling (Already Implemented)

[prepare_data.py](file:///sites/org_vs_person/scripts/prepare_data.py) uses stratified train/test split:

```python
train_df, test_df = train_test_split(
    df_clean, test_size=test_size, 
    stratify=df_clean["type"],  # Maintains 6:1 ratio
    random_state=random_state
)
```

This ensures both train and test sets have the same 6:1 ratio.

### 2. ✅ Balanced Class Weights (JUST ADDED)

**Most Important Fix!**

Updated [train_model.py](file:///sites/org_vs_person/scripts/train_model.py) to use `class_weight='balanced'`:

```python
"LogisticRegression": LogisticRegression(
    max_iter=1000, 
    random_state=42, 
    class_weight="balanced"  # ⬅️ THIS IS KEY!
),
```

**How it works:**
- Sklearn automatically computes weight for each class: `n_samples / (n_classes * class_count)`
- For our 6:1 ratio:
  - **ORG weight:** ~6.0 (minority class gets higher weight)
  - **PER weight:** ~1.0 (majority class gets standard weight)
- During training, misclassifying an ORG is penalized 6x more heavily

**Applied to:**
- ✅ LogisticRegression
- ✅ LinearSVC
- ✅ RandomForest
- ❌ MultinomialNB (doesn't support class_weight parameter)

### 3. ✅ Weighted F1 Score (Already Implemented)

Using `f1_score(average='weighted')` accounts for class imbalance in evaluation:
- Weights each class's F1 by its support (number of samples)
- Gives a more realistic performance metric than raw accuracy

### 4. ✅ Per-Class F1 Monitoring (JUST ADDED)

Enhanced evaluation to track **ORG F1** and **PER F1** separately:

```python
# Per-class F1 scores to monitor class imbalance handling
test_f1_per_class = f1_score(y_test, y_pred, average=None, labels=["ORG", "PER"])

return {
    "test_f1_org": test_f1_per_class[0],  # ORG F1
    "test_f1_per": test_f1_per_class[1],  # PER F1
    ...
}
```

**Benefits:**
- You'll see if ORG classification is suffering
- Can compare ORG vs PER performance explicitly
- Example output:
  ```
  Test F1 (weighted): 0.9200
  Test F1 ORG: 0.8500  ⬅️ Can monitor minority class
  Test F1 PER: 0.9350  ⬅️ vs majority class
  ```

## Alternative Strategies (Not Implemented)

If you want even better ORG performance, consider:

### Option A: Resampling

**Undersample PER:**
```python
# In prepare_data.py, after deduplication:
per_df = df_processed[df_processed['type'] == 'PER']
org_df = df_processed[df_processed['type'] == 'ORG']

# Match PER count to ORG count
per_sampled = per_df.sample(n=len(org_df), random_state=42)
df_balanced = pd.concat([per_sampled, org_df])
```

**Oversample ORG:**
```python
from sklearn.utils import resample

org_upsampled = resample(org_df, n_samples=len(per_df), random_state=42)
df_balanced = pd.concat([org_upsampled, per_df])
```

**Trade-offs:**
- Undersampling: Loses data but speeds up training
- Oversampling: Risk of overfitting duplicate ORG examples

### Option B: SMOTE (Synthetic Minority Over-sampling)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

Generates synthetic ORG examples in feature space.

### Option C: Custom Decision Threshold

```python
# Instead of default 0.5, optimize threshold for ORG
probas = classifier.predict_proba(X_test)
# Adjust threshold to favor ORG predictions
predictions = (probas[:, org_index] > 0.3).astype(int)
```

## What to Expect

With `class_weight='balanced'`:

**Before:**
- Accuracy: 85% (just predicting PER all the time!)
- PER F1: 0.92
- ORG F1: 0.45 😞

**After:**
- Accuracy: ~88-92% (more balanced predictions)
- PER F1: 0.91-0.93
- ORG F1: 0.75-0.85 ✅ Much better!

## Verification

When you train the model, watch for:
1. **ORG F1 score > 0.70** - Good minority class performance
2. **Similar F1 scores** - ORG and PER F1 should be within ~10% of each other
3. **Classification report** - Shows precision/recall for both classes

If ORG F1 is still too low (<0.60), consider the resampling options above.

## Summary

✅ **Class imbalance IS a concern** - Your instinct was correct!  
✅ **We've handled it** - Using balanced class weights  
✅ **Monitoring in place** - Per-class F1 scores will show effectiveness  
✅ **Further options available** - Can try resampling if needed  

The `class_weight='balanced'` parameter is the industry-standard approach and should give good results for our 6:1 imbalance.
