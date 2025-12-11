"""Unit tests for data processing functions."""
import pytest
from pathlib import Path
import pandas as pd
import sys

# Add scripts directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_data import filter_and_deduplicate, create_train_test_split, sample_data


class TestFilterAndDeduplicate:
    """Tests for filter_and_deduplicate function."""

    def test_filters_loc_entries(self):
        """Test that LOC entries are filtered out."""
        df = pd.DataFrame(
            {
                "wikidata_id": ["Q1", "Q2", "Q3", "Q4"],
                "eng": ["A", "B", "C", "D"],
                "label": ["A", "B", "C", "D"],
                "language": ["en", "en", "en", "en"],
                "type": ["PER", "ORG", "LOC", "PER"],
            }
        )

        result = filter_and_deduplicate(df)

        assert len(result) == 3  # LOC should be filtered out
        assert "LOC" not in result["type"].values
        assert set(result["type"].unique()) == {"PER", "ORG"}

    def test_keeps_per_and_org(self):
        """Test that PER and ORG entries are kept."""
        df = pd.DataFrame(
            {
                "wikidata_id": ["Q1", "Q2"],
                "eng": ["Person Name", "Org Name"],
                "label": ["Person Name", "Org Name"],
                "language": ["en", "en"],
                "type": ["PER", "ORG"],
            }
        )

        result = filter_and_deduplicate(df)

        assert len(result) == 2
        assert set(result["type"].values) == {"PER", "ORG"}

    def test_deduplicates_by_label(self):
        """Test that duplicate labels are removed."""
        df = pd.DataFrame(
            {
                "wikidata_id": ["Q1", "Q1", "Q1", "Q2"],
                "eng": ["Name", "Name", "Name", "Other"],
                "label": ["Name", "Name", "Name", "Other"],
                "language": ["en", "fr", "de", "en"],
                "type": ["PER", "PER", "PER", "ORG"],
            }
        )

        result = filter_and_deduplicate(df)

        assert len(result) == 2  # Only 2 unique labels
        assert result["label"].nunique() == 2

    def test_keeps_first_occurrence(self):
        """Test that first occurrence is kept when deduplicating."""
        df = pd.DataFrame(
            {
                "wikidata_id": ["Q1", "Q1"],
                "eng": ["Name", "Name"],
                "label": ["Test Name", "Test Name"],
                "language": ["en", "fr"],
                "type": ["PER", "PER"],
            }
        )

        result = filter_and_deduplicate(df)

        assert len(result) == 1
        # First occurrence (English) should be kept
        assert result.iloc[0]["language"] == "en"


class TestCreateTrainTestSplit:
    """Tests for create_train_test_split function."""

    def test_split_proportions(self):
        """Test that split proportions are approximately correct."""
        df = pd.DataFrame(
            {"label": [f"Name_{i}" for i in range(100)], "type": ["PER"] * 50 + ["ORG"] * 50}
        )

        train_df, test_df = create_train_test_split(df, test_size=0.2)

        assert len(train_df) == 80
        assert len(test_df) == 20

    def test_stratified_split(self):
        """Test that split maintains class balance (stratified)."""
        df = pd.DataFrame(
            {"label": [f"Name_{i}" for i in range(100)], "type": ["PER"] * 70 + ["ORG"] * 30}
        )

        train_df, test_df = create_train_test_split(df, test_size=0.2)

        # Check class proportions in train set
        train_per_ratio = (train_df["type"] == "PER").sum() / len(train_df)
        assert abs(train_per_ratio - 0.7) < 0.05  # Should be close to 70%

        # Check class proportions in test set
        test_per_ratio = (test_df["type"] == "PER").sum() / len(test_df)
        assert abs(test_per_ratio - 0.7) < 0.1  # Should be close to 70%

    def test_no_overlap(self):
        """Test that train and test sets don't overlap."""
        df = pd.DataFrame(
            {"label": [f"Name_{i}" for i in range(100)], "type": ["PER"] * 50 + ["ORG"] * 50}
        )

        train_df, test_df = create_train_test_split(df)

        # Check no overlap in labels
        train_labels = set(train_df["label"])
        test_labels = set(test_df["label"])

        assert len(train_labels.intersection(test_labels)) == 0

    def test_reproducibility(self):
        """Test that split is reproducible with same random_state."""
        df = pd.DataFrame(
            {"label": [f"Name_{i}" for i in range(100)], "type": ["PER"] * 50 + ["ORG"] * 50}
        )

        train1, test1 = create_train_test_split(df, random_state=42)
        train2, test2 = create_train_test_split(df, random_state=42)

        assert train1.equals(train2)
        assert test1.equals(test2)


class TestSampleData:
    """Tests for sample_data function."""

    def test_sample_size(self):
        """Test that sample has correct size."""
        df = pd.DataFrame(
            {"label": [f"Name_{i}" for i in range(1000)], "type": ["PER"] * 700 + ["ORG"] * 300}
        )

        result = sample_data(df, sample_size=100)

        assert len(result) == 100

    def test_maintains_class_balance(self):
        """Test that sampling maintains approximate class balance."""
        df = pd.DataFrame(
            {"label": [f"Name_{i}" for i in range(1000)], "type": ["PER"] * 700 + ["ORG"] * 300}
        )

        result = sample_data(df, sample_size=100)

        per_ratio = (result["type"] == "PER").sum() / len(result)
        # Should be close to 70% (original ratio)
        assert abs(per_ratio - 0.7) < 0.15

    def test_sample_larger_than_data(self):
        """Test behavior when sample size >= data size."""
        df = pd.DataFrame({"label": ["A", "B", "C"], "type": ["PER", "ORG", "PER"]})

        result = sample_data(df, sample_size=100)

        # Should return full dataset
        assert len(result) == 3

    def test_reproducibility(self):
        """Test that sampling is reproducible with same random_state."""
        df = pd.DataFrame(
            {"label": [f"Name_{i}" for i in range(1000)], "type": ["PER"] * 700 + ["ORG"] * 300}
        )

        result1 = sample_data(df, sample_size=100, random_state=42)
        result2 = sample_data(df, sample_size=100, random_state=42)

        assert result1.equals(result2)
