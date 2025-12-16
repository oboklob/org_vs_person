#!/usr/bin/env python
"""Quick diagnostic to test feature impact on SGDClassifier performance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

print("=" * 80)
print("PERFORMANCE DIAGNOSTIC")
print("=" * 80)

# Load a sample of data
data_dir = Path("data")
train_df = pd.read_csv(data_dir / "train.csv").sample(n=100000, random_state=42)
test_df = pd.read_csv(data_dir / "test.csv").sample(n=25000, random_state=42)

X_train_text = train_df["label"].values
y_train = train_df["type"].values
X_test_text = test_df["label"].values
y_test = test_df["type"].values

print(f"Train samples: {len(X_train_text):,}")
print(f"Test samples: {len(X_test_text):,}")
print()

# Test 1: Baseline (your old approach)
print("Test 1: Baseline n-grams only (2^18 features)")
print("-" * 80)
vec1 = HashingVectorizer(analyzer='char', ngram_range=(3, 5), n_features=2**18)
X_train_1 = vec1.fit_transform(X_train_text)
X_test_1 = vec1.transform(X_test_text)

clf1 = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42, class_weight='balanced')
clf1.fit(X_train_1, y_train)
y_pred_1 = clf1.predict(X_test_1)

acc1 = accuracy_score(y_test, y_pred_1)
f1_1 = f1_score(y_test, y_pred_1, average='weighted')
print(f"  Accuracy: {acc1:.4f}")
print(f"  F1 Score: {f1_1:.4f}")
print(f"  Feature count: {X_train_1.shape[1]:,}")
print()

# Test 2: With engineered features (no dropout)
print("Test 2: N-grams + Engineered features (NO dropout)")
print("-" * 80)
from sklearn.pipeline import FeatureUnion, Pipeline
from name_classifier.transformers import NameFeatureExtractor, TextExtractor

char_ngrams = Pipeline([
    ('text_extract', TextExtractor()),
    ('hashing', HashingVectorizer(analyzer='char', ngram_range=(3, 5), n_features=2**18))
])

feature_union = FeatureUnion([
    ('char_ngrams', char_ngrams),
    ('engineered', NameFeatureExtractor())
])

# Need DataFrame for FeatureUnion
X_train_df_2 = pd.DataFrame({'label': X_train_text})
X_test_df_2 = pd.DataFrame({'label': X_test_text})

X_train_2 = feature_union.fit_transform(X_train_df_2)
X_test_2 = feature_union.transform(X_test_df_2)

clf2 = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42, class_weight='balanced')
clf2.fit(X_train_2, y_train)
y_pred_2 = clf2.predict(X_test_2)

acc2 = accuracy_score(y_test, y_pred_2)
f1_2 = f1_score(y_test, y_pred_2, average='weighted')
print(f"  Accuracy: {acc2:.4f}")
print(f"  F1 Score: {f1_2:.4f}")
print(f"  Feature count: {X_train_2.shape[1]:,}")
print()

# Test 3: Tuned SGDClassifier
print("Test 3: Baseline with TUNED SGDClassifier")
print("-" * 80)
clf3 = SGDClassifier(
    loss='log_loss',
    max_iter=2000,
    random_state=42,
    class_weight='balanced',
    alpha=0.0001,
    learning_rate='optimal',
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5
)
clf3.fit(X_train_1, y_train)
y_pred_3 = clf3.predict(X_test_1)

acc3 = accuracy_score(y_test, y_pred_3)
f1_3 = f1_score(y_test, y_pred_3, average='weighted')
print(f"  Accuracy: {acc3:.4f}")
print(f"  F1 Score: {f1_3:.4f}")
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Baseline (old):                    Acc={acc1:.4f}, F1={f1_1:.4f}")
print(f"With engineered features:          Acc={acc2:.4f}, F1={f1_2:.4f}")
print(f"With tuned SGD:                    Acc={acc3:.4f}, F1={f1_3:.4f}")
print()
print("Recommendations:")
if acc2 < acc1:
    print("  ⚠️  Engineered features are HURTING performance!")
    print("  → Try training without them or with less aggressive dropout")
if acc3 > acc1:
    print("  ✅ Tuned SGD hyperparameters help!")
if acc1 > 0.92:
    print("  ✅ Baseline performance is good on this sample")
