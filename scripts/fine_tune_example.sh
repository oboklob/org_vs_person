#!/bin/bash

# Fine-tune LinearSVC with the best vectorizer
# This will test 16 combinations (8 C values × 2 loss functions)

python scripts/train_model.py \
  --fix-vectorizer hash_char_3-5_256k \
  --fix-model LinearSVC_fine \
  --data-dir data \
  --output-dir name_classifier/models \
  --n-jobs -1
