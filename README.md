# Name Classifier

A machine learning-based Python package for classifying names as either **organizations (ORG)** or **individuals (PER)**.

The classifier is trained on the [paranames dataset](https://github.com/ooglyBoogly/para-names) and uses scikit-learn for classification.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from name_classifier import classify, classify_list

# Classify individual names
result = classify("Bob Smith")
print(result)  # Output: PER

# Classify organization names
result = classify("Microsoft Corporation")
print(result)  # Output: ORG

# Classify multiple names at once
results = classify_list(['Bob Smith', 'Google Inc.', 'ministry of defense'])
print(results)  # Output: ['PER', 'ORG', 'ORG']
```

## Advanced Usage

```python
from name_classifier import NameClassifier

# Create classifier instance
classifier = NameClassifier()

# Classify individual names
names = ["John Doe", "Apple Inc", "Jane Williams", "Google LLC"]
for name in names:
    result = classifier.classify(name)
    print(f"{name}: {result}")

# Classify multiple names efficiently (batch processing)
names = ["John Doe", "Apple Inc", "Jane Williams", "Google LLC"]
results = classifier.classify_list(names)
for name, result in zip(names, results):
    print(f"{name}: {result}")
```

## Development

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### Training the Model

1. **Prepare the data:**
   ```bash
   # Full dataset with caching (recommended)
   python scripts/prepare_data.py
   
   # Sample dataset for quick experimentation
   python scripts/prepare_data.py --sample-size 100000
   
   # Force rebuild cache
   python scripts/prepare_data.py --force-rebuild
   
   # Disable caching
   python scripts/prepare_data.py --no-cache
   ```

2. **Train and evaluate models:**
   ```bash
   # Fast training with pre-configured best model (recommended for iteration)
   python scripts/train_model.py --fast-train
   
   # Full grid search with multiprocessing
   python scripts/train_model.py --n-jobs -1
   
   # Fast training with sample data (quickest iteration)
   python scripts/prepare_data.py --sample-size 50000
   python scripts/train_model.py --fast-train
   ```

**Performance Tips:**
- Use `--fast-train` to skip grid search and train only the best-performing model (TF-IDF char 2-4 + LogisticRegression)
- Use `--use-cache` (enabled by default) to cache the deduped dataset and avoid rebuilding from scratch
- Use `--n-jobs -1` to enable multiprocessing with all available CPU cores
- For quick experimentation, use `--sample-size 50000` with `--fast-train`

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=name_classifier --cov-report=html

# Run specific test file
pytest tests/test_classifier.py -v
```

## Model Information

The classifier uses a combination of:
- **Vectorization**: Character and word n-grams (TF-IDF)
- **Algorithm**: Selected from multiple classifiers (Logistic Regression, Linear SVM, Random Forest, Naive Bayes)
- **Training Data**: Paranames dataset with ~140M name-entity pairs

The best performing model is automatically selected based on cross-validation results.

## License

MIT License

## Data Source

This package uses the [paranames dataset](https://github.com/ooglyBoogly/para-names), which aggregates named entity data from Wikidata.
