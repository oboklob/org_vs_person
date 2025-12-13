"""Text normalization utilities for name classification.

This module provides shared normalization functions used by both
ISO 20275 legal form matching and feature engineering.
"""
import re
import unicodedata
from dataclasses import dataclass
from typing import List


@dataclass
class NormalizedText:
    """Container for normalized text and its components.
    
    Attributes:
        raw_str: Original input string
        norm_str: Normalized string (casefolded, punctuation replaced, whitespace collapsed)
        tokens: List of tokens from splitting normalized string on whitespace
    """
    raw_str: str
    norm_str: str
    tokens: List[str]


def strip_diacritics(text: str) -> str:
    """Remove diacritical marks from text.
    
    Args:
        text: Input text with potential diacritics
        
    Returns:
        Text with diacritics removed (e.g., 'café' -> 'cafe')
    """
    # Normalize to NFD (decomposed form) then filter out combining marks
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')


def normalize(text: str, strip_diacritics_flag: bool = False) -> NormalizedText:
    """Normalize text for consistent processing.
    
    Performs the following transformations:
    1. Casefold (lowercase)
    2. Optionally strip diacritics
    3. Replace punctuation with spaces
    4. Collapse multiple spaces
    5. Trim leading/trailing whitespace
    6. Tokenize on spaces
    
    Args:
        text: Input text to normalize
        strip_diacritics_flag: If True, remove diacritical marks
        
    Returns:
        NormalizedText object with raw_str, norm_str, and tokens
        
    Example:
        >>> result = normalize("Café & Co., Ltd.")
        >>> result.norm_str
        'cafe co ltd'
        >>> result.tokens
        ['cafe', 'co', 'ltd']
    """
    if text is None:
        text = ""
    
    raw_str = str(text)
    
    # Step 1: Casefold
    normalized = raw_str.lower()
    
    # Step 2: Optionally strip diacritics
    if strip_diacritics_flag:
        normalized = strip_diacritics(normalized)
    
    # Step 3: Replace punctuation with spaces
    # Keep alphanumeric and spaces, replace everything else with space
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    
    # Step 4: Collapse multiple spaces and trim
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Step 5: Tokenize on spaces
    tokens = normalized.split() if normalized else []
    
    return NormalizedText(
        raw_str=raw_str,
        norm_str=normalized,
        tokens=tokens
    )
