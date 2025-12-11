# Name Classifier

A machine learning-based Python package for classifying names as either **organizations (ORG)** or **individuals (PER)**.

The classifier is trained on the [paranames dataset](https://github.com/ooglyBoogly/para-names) and uses scikit-learn for classification.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from name_classifier import classify

# Classify individual names
result = classify("Bob Smith")
print(result)  # Output: PER

# Classify organization names
result = classify("Microsoft Corporation")
print(result)  # Output: ORG
```

## Advanced Usage

```python
from name_classifier import NameClassifier

# Create classifier instance
classifier = NameClassifier()

# Classify multiple names
names = ["John Doe", "Apple Inc", "Jane Williams", "Google LLC"]
for name in names:
    result = classifier.classify(name)
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
   python scripts/prepare_data.py --sample-size 1000000
   ```

2. **Train and evaluate models:**
   ```bash
   python scripts/train_model.py
   ```

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
