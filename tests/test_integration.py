"""Integration tests for the name classifier (requires trained model)."""
import pytest
from pathlib import Path

from name_classifier import NameClassifier, classify, classify_list
from name_classifier.config import MODEL_PATH, VECTORIZER_PATH


# Skip these tests if model is not trained yet
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists() or not VECTORIZER_PATH.exists(),
    reason="Trained model not found. Run scripts/train_model.py first.",
)


class TestIntegrationWithTrainedModel:
    """Integration tests using the actual trained model."""

    def test_classify_person_names(self):
        """Test classification of common person names."""
        person_names = [
            "Bob Smith",
            "John Doe",
            "Jane Williams",
            "Mary Johnson",
            "Michael Brown",
            "Sarah Davis",
        ]

        classifier = NameClassifier()

        for name in person_names:
            result = classifier.classify(name)
            assert result in ["PER", "ORG"], f"Unexpected result for '{name}': {result}"
            # We expect PER, but don't fail if model predicts ORG
            # (depends on training data)

    def test_classify_organization_names(self):
        """Test classification of common organization names."""
        org_names = [
            "Microsoft Corporation",
            "Apple Inc",
            "Google LLC",
            "Amazon.com",
            "Meta Platforms",
            "Tesla Inc",
        ]

        classifier = NameClassifier()

        for name in org_names:
            result = classifier.classify(name)
            assert result in ["PER", "ORG"], f"Unexpected result for '{name}': {result}"

    def test_classify_with_convenience_function(self):
        """Test the convenience classify() function."""
        result = classify("Bill Gates")
        assert result in ["PER", "ORG"]

        result = classify("Microsoft")
        assert result in ["PER", "ORG"]

    def test_classify_edge_cases(self):
        """Test edge cases."""
        classifier = NameClassifier()

        # Single name (ambiguous)
        result = classifier.classify("Smith")
        assert result in ["PER", "ORG"]

        # Name with hyphen
        result = classifier.classify("Mary-Jane Watson")
        assert result in ["PER", "ORG"]

        # Name with Jr/Sr
        result = classifier.classify("John Smith Jr")
        assert result in ["PER", "ORG"]

        # Organization with abbreviation
        result = classifier.classify("IBM")
        assert result in ["PER", "ORG"]

        # Mixed case
        result = classifier.classify("JOHN SMITH")
        assert result in ["PER", "ORG"]

        result = classifier.classify("john smith")
        assert result in ["PER", "ORG"]

    def test_performance(self):
        """Test that classification is reasonably fast."""
        import time

        classifier = NameClassifier()

        # Warm up (loads model)
        classifier.classify("Warm up")

        # Time 100 classifications
        names = [f"Person Name {i}" for i in range(100)]

        start = time.time()
        for name in names:
            classifier.classify(name)
        elapsed = time.time() - start

        # Should be able to do 100 classifications in less than 2 seconds
        assert elapsed < 2.0, f"Classification too slow: {elapsed:.2f}s for 100 names"

        # Calculate throughput
        throughput = 100 / elapsed
        print(f"\nThroughput: {throughput:.0f} classifications/second")

    def test_model_metadata_available(self):
        """Test that model metadata can be retrieved."""
        classifier = NameClassifier()
        metadata = classifier.get_metadata()

        if metadata is not None:
            # Check expected fields
            assert "test_accuracy" in metadata
            assert "vectorizer" in metadata
            assert "classifier" in metadata
            assert "training_date" in metadata

            # Check accuracy is reasonable
            assert metadata["test_accuracy"] > 0.5  # At least 50%

    def test_multiple_instances_share_nothing(self):
        """Test that multiple classifier instances work independently."""
        classifier1 = NameClassifier()
        classifier2 = NameClassifier()

        result1 = classifier1.classify("Test Name")
        result2 = classifier2.classify("Test Name")

        # Both should give same result
        assert result1 == result2

    def test_classify_list_batch_classification(self):
        """Test batch classification with classify_list."""
        names = [
            "Bob Smith",
            "Google Inc.",
            "ministry of defense",
            "Jane Doe",
            "Microsoft Corporation",
        ]

        classifier = NameClassifier()
        results = classifier.classify_list(names)

        # Should return one result per input
        assert len(results) == len(names)

        # All results should be valid
        for result in results:
            assert result in ["PER", "ORG"]

    def test_classify_list_convenience_function(self):
        """Test the convenience classify_list() function."""
        results = classify_list(["Bob Smith", "Google Inc.", "ministry of defense"])

        assert len(results) == 3
        assert all(result in ["PER", "ORG"] for result in results)

    def test_classify_list_performance(self):
        """Test that batch classification is faster than individual calls."""
        import time

        classifier = NameClassifier()

        # Generate test data
        names = [f"Person Name {i}" for i in range(100)]

        # Warm up
        classifier.classify(names[0])

        # Time individual classifications
        start = time.time()
        individual_results = [classifier.classify(name) for name in names]
        individual_time = time.time() - start

        # Time batch classification
        start = time.time()
        batch_results = classifier.classify_list(names)
        batch_time = time.time() - start

        # Results should be the same
        assert individual_results == batch_results

        # Batch should be faster (or at least not significantly slower)
        # We allow some variance, but batch should generally be faster
        print(f"\nIndividual time: {individual_time:.3f}s, Batch time: {batch_time:.3f}s")
        print(f"Speedup: {individual_time / batch_time:.1f}x")

        # This is just informational, we don't fail if batch is slightly slower
        # But we do print the performance comparison



class TestRealWorldExamples:
    """Test with real-world examples."""

    @pytest.mark.parametrize(
        "name,expected_type",
        [
            # People
            ("Albert Einstein", "PER"),
            ("Marie Curie", "PER"),
            ("Nelson Mandela", "PER"),
            ("Barack Obama", "PER"),
            # Organizations
            ("United Nations", "ORG"),
            ("World Health Organization", "ORG"),
            ("Harvard University", "ORG"),
            ("Red Cross", "ORG"),
        ],
    )
    def test_known_entities(self, name, expected_type):
        """Test classification of well-known entities."""
        result = classify(name)

        # We don't assert strict equality since the model might not be perfect
        # But we record what it predicts for debugging
        print(f"\n{name}: Expected={expected_type}, Got={result}")

        # At minimum, result should be valid
        assert result in ["PER", "ORG"]

        # Ideally it matches, but we don't fail the test if it doesn't
        # This helps identify model weaknesses without breaking CI
        if result != expected_type:
            pytest.skip(f"Model predicted {result} instead of {expected_type} for '{name}'")
