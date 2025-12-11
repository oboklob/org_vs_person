#!/usr/bin/env python3
"""
Demonstration of the classify_list function for batch name classification.

This script shows how to use the classify_list function to classify
multiple names efficiently in a single call.
"""

from name_classifier import classify, classify_list


def main():
    """Demonstrate batch classification with classify_list."""
    print("Name Classifier - Batch Classification Demo")
    print("=" * 50)
    print()

    # Example from the user's request
    print("Example 1: User's request")
    print("-" * 50)
    names = ['Bob Smith', 'Google Inc.', 'ministry of defense']
    results = classify_list(names)
    print(f"Input: {names}")
    print(f"Output: {results}")
    print()

    # More comprehensive example
    print("Example 2: Mixed person and organization names")
    print("-" * 50)
    names = [
        "John Doe",
        "Microsoft Corporation",
        "Jane Williams",
        "Apple Inc",
        "United Nations",
        "Albert Einstein",
        "Harvard University",
        "Marie Curie"
    ]
    
    results = classify_list(names)
    
    print(f"{'Name':<30} {'Classification':<15}")
    print("-" * 45)
    for name, result in zip(names, results):
        print(f"{name:<30} {result:<15}")
    print()

    # Performance comparison
    print("Example 3: Performance comparison")
    print("-" * 50)
    import time

    # Generate test data
    test_names = [f"Person Name {i}" for i in range(100)]

    # Individual classification
    start = time.time()
    individual_results = [classify(name) for name in test_names]
    individual_time = time.time() - start

    # Batch classification
    start = time.time()
    batch_results = [classify_list(test_names)]
    batch_time = time.time() - start

    print(f"Classifying 100 names:")
    print(f"  Individual calls: {individual_time:.3f}s")
    print(f"  Batch call:       {batch_time:.3f}s")
    if batch_time > 0:
        print(f"  Speedup:          {individual_time / batch_time:.1f}x")
    print()


if __name__ == "__main__":
    main()
