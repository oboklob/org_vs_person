#!/usr/bin/env python
"""Check if model classes are in expected order."""
import joblib
import sys
from pathlib import Path

model = joblib.load("name_classifier/models/model.pkl")

print("Model classes_:", model.classes_)
print()
print("Expected: ['ORG', 'PER'] or ['PER', 'ORG']")
print()

if hasattr(model, 'classes_'):
    classes = model.classes_
    print(f"Class at index 0: {classes[0]}")
    print(f"Class at index 1: {classes[1]}")
    print()
    
    # Check coefficient interpretation
    coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
    
    print("Coefficient interpretation:")
    print(f"  Positive coefficient → predicts class at index 1: {classes[1]}")
    print(f"  Negative coefficient → predicts class at index 0: {classes[0]}")
    print()
    
    if classes[0] == 'PER' and classes[1] == 'ORG':
        print("✅ CORRECT: Positive coef → ORG, Negative coef → PER")
        print("   This means feature signs in analysis are CORRECT")
    elif classes[0] == 'ORG' and classes[1] == 'PER':
        print("❌ INVERTED: Positive coef → PER, Negative coef → ORG")
        print("   This means feature signs in analysis are BACKWARDS!")
        print("   The model IS working correctly, the ANALYSIS is wrong!")
