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

## Confidence-Based Classification

For cases where you need to know **how confident** the model is, or want to handle uncertain classifications explicitly:

```python
from name_classifier import classify_with_confidence, filter_by_confidence

# Get classification with confidence score
label, confidence = classify_with_confidence("Bob Smith", min_confidence=0.8)
if label == "UNCERTAIN":
    print(f"Not confident enough (confidence: {confidence:.2%})")
else:
    print(f"Classified as {label} with {confidence:.2%} confidence")

# Filter a list to get only names you're confident about
names = ['Bob Smith', 'Google Inc.', 'Jane Doe', 'Apple Inc', 'Jordan']
certain_persons = filter_by_confidence(names, "PER", min_confidence=0.85)
print("Confident persons:")
for name, conf in certain_persons:
    print(f"  {name}: {conf:.2%} confident")

# Output:
# Confident persons:
#   Bob Smith: 100.00% confident
#   Jane Doe: 100.00% confident
```

**When to use confidence-based classification:**
- **Data quality**: Filter names to include only high-confidence classifications
- **Uncertainty handling**: Identify ambiguous names that need manual review
- **Threshold control**: Adjust `min_confidence` based on your precision/recall needs

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

#### Corpus Configuration (Optional)

The data preparation script supports loading data from multiple corpus files with different formats using a YAML configuration file. This is useful when you want to combine multiple data sources (e.g., paranames dataset + company registers + custom name lists).

**Configuration Schema:**

Create a `corpus/corpus_config.yaml` file (see `corpus/corpus_config.example.yaml` for a complete example):

```yaml
sources:
  - name: "paranames"
    path: "corpus/paranames.tsv.gz"
    format: "tsv"                    # "tsv" or "csv"
    compression: "gzip"              # "gzip" or null
    columns:
      name: "label"                  # Column with names
      classification: "type"         # Column with ORG/PER classification
    filters:
      type: ["ORG", "PER"]          # Optional: filter to specific types
  
  - name: "company_register"
    path: "corpus/organizations.csv"
    format: "csv"
    compression: null
    columns:
      name: "organization_name"
      classification: null           # No classification column
    fixed_classification: "ORG"     # Apply ORG to all rows
```

Each source supports:
- Different file formats (TSV, CSV)
- Different compression (gzip or uncompressed)
- Column mapping to standardize data
- Fixed classification when source has no classification column
- Filters to include only specific types

**How it works:**
- If `corpus/corpus_config.yaml` exists, the script loads and merges all configured sources
- If no config exists, it falls back to loading only `corpus/paranames.tsv.gz` (legacy mode)
- The cache is automatically invalidated when the configuration changes

#### Preparing the Data

1. **Prepare the data:**
   ```bash
   # With configuration (recommended for multiple sources)
   python scripts/prepare_data.py
   
   # With custom config path
   python scripts/prepare_data.py --config-path my_config.yaml
   
   # Keep only Latin-script names (filters out Japanese, Chinese, Arabic, etc.)
   python scripts/prepare_data.py --latin-only
   
   # Sample dataset for quick experimentation
   python scripts/prepare_data.py --sample-size 100000
   
   # Force rebuild cache
   python scripts/prepare_data.py --force-rebuild
   
   # Disable caching
   python scripts/prepare_data.py --no-cache
   ```

   **Data Cleaning:**
   The script automatically cleans the data by:
   - Removing rows with null/empty labels
   - Removing whitespace-only labels
   - Optionally filtering to Latin-only characters with `--latin-only` (removes non-transliterated names in Japanese, Chinese, Cyrillic, Arabic, etc.)

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
