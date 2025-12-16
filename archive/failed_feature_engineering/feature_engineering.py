"""Feature engineering for name classification.

Extracts 32 engineered features from names, combining ISO 20275 legal forms,
language/jurisdiction hints, string structure, casing patterns, and lexical signals.
"""
import re
from typing import Optional, List
import numpy as np

from name_classifier.normalization import normalize
from name_classifier.iso20275_matcher import ISO20275Matcher


# Language/jurisdiction bucketing
LANGUAGE_BUCKETS = {
    'en_like': {'en', 'gb', 'us', 'au', 'ca', 'ie', 'nz'},
    'germanic': {'de', 'at', 'ch', 'nl', 'be'},
    'romance': {'fr', 'es', 'pt', 'it', 'ro'},
    'nordic': {'se', 'no', 'dk', 'fi', 'is'}
}

# Person title patterns
PERSON_TITLES = {
    'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'professor',
    'sir', 'dame', 'lord', 'lady', 'rev', 'reverend'
}

# Person suffix patterns
PERSON_SUFFIXES = {
    'jr', 'sr', 'ii', 'iii', 'iv', 'phd', 'md', 'esq', 'esquire'
}

# Organization keywords
ORG_KEYWORDS = {
    'foundation', 'trust', 'university', 'college', 'council', 'ministry',
    'association', 'club', 'company', 'group', 'partners', 'limited',
    'corporation', 'inc', 'incorporated', 'llc', 'corp', 'plc'
}

# Common given names (top 50)
GIVEN_NAMES = {
    'john', 'mary', 'david', 'sarah', 'michael', 'jennifer', 'james', 'linda',
    'robert', 'patricia', 'william', 'elizabeth', 'richard', 'susan', 'joseph',
    'jessica', 'thomas', 'karen', 'charles', 'nancy', 'christopher', 'margaret',
    'daniel', 'lisa', 'matthew', 'betty', 'anthony', 'dorothy', 'mark', 'sandra',
    'donald', 'ashley', 'steven', 'kimberly', 'paul', 'donna', 'andrew', 'emily',
    'joshua', 'carol', 'kenneth', 'michelle', 'kevin', 'amanda', 'brian', 'melissa',
    'george', 'deborah', 'edward', 'stephanie'
}

# Surname particles
SURNAME_PARTICLES = {
    'van', 'von', 'de', 'del', 'della', 'da', 'dos', 'das',
    'mac', 'mc', "o'", 'o', 'le', 'la', 'di', 'du'
}


def bucket_language(language_code: Optional[str]) -> str:
    """Bucket a language code into coarse categories.
    
    Args:
        language_code: ISO 639-1 language code (e.g., 'en', 'de')
        
    Returns:
        Bucket name: 'en_like', 'germanic', 'romance', 'nordic', 'other', or 'unknown'
    """
    if not language_code:
        return 'unknown'
    
    lang = language_code.lower().strip()
    
    for bucket_name, codes in LANGUAGE_BUCKETS.items():
        if lang in codes:
            return bucket_name
    
    return 'other'


def extract_iso_features(name: str, iso_matcher: ISO20275Matcher) -> np.ndarray:
    """Extract ISO 20275-derived features (8 features).
    
    Args:
        name: Name to extract features from
        iso_matcher: ISO20275Matcher instance
        
    Returns:
        Array of 8 features
    """
    features = np.zeros(8, dtype=np.float32)
    
    match = iso_matcher.match_legal_form(name)
    
    if match:
        features[0] = 1.0  # has_legal_form
        
        # Tier one-hot (indices 1-3)
        if match.metadata.tier == 'A':
            features[1] = 1.0
        elif match.metadata.tier == 'B':
            features[2] = 1.0
        elif match.metadata.tier == 'C':
            features[3] = 1.0
        
        features[4] = float(match.token_len)  # legal_form_token_len
        features[5] = float(match.char_len)   # legal_form_char_len
        features[6] = 1.0 if match.is_ambiguous_short else 0.0
        features[7] = 1.0 if match.metadata.country else 0.0  # country_is_known
    
    return features


def extract_language_jurisdiction_features(match, language_hint, jurisdiction_hint):
    """Extract language and jurisdiction hint features.
    
    NOTE: These features are currently REMOVED as they were unused (all zeros).
    Keeping this function as a stub in case we want to add them back later
    when language/jurisdiction data becomes available.
    
    Args:
        match: SuffixMatch or None
        language_hint: Optional language code hint
        jurisdiction_hint: Optional jurisdiction code hint
        
    Returns:
        Empty array (was 8 features, now 0)
    """
    # REMOVED: These were always zero in training data
    # - has_language_hint
    # - has_jurisdiction_hint  
    # - language_bucket_en_like, germanic, romance, nordic, other
    # - jurisdiction_reserved
    
    return np.array([], dtype=np.float32)


def extract_string_structure_features(name: str, raw_str: str) -> np.ndarray:
    """Extract string structure features (8 features).
    
    Args:
        name: Normalized name
        raw_str: Original raw string (for punctuation detection)
        
    Returns:
        Array of 8 features
    """
    features = np.zeros(8, dtype=np.float32)
    
    normalized = normalize(name)
    
    # Basic measurements
    features[0] = float(len(normalized.norm_str))  # char_len
    features[1] = float(len(normalized.tokens))     # token_count
    
    # Average token length
    if normalized.tokens:
        avg_token_len = sum(len(t) for t in normalized.tokens) / len(normalized.tokens)
        features[2] = avg_token_len
    
    # Binary indicators (check raw string for these)
    features[3] = 1.0 if any(c.isdigit() for c in raw_str) else 0.0  # contains_digit
    features[4] = 1.0 if '&' in raw_str else 0.0                     # contains_ampersand
    features[5] = 1.0 if ',' in raw_str else 0.0                     # contains_comma
    features[6] = 1.0 if ('/' in raw_str or '-' in raw_str) else 0.0  # contains_slash_or_hyphen
    features[7] = 1.0 if '.' in raw_str else 0.0                     # contains_period
    
    return features


def extract_casing_initials_features(raw_str: str) -> np.ndarray:
    """Extract casing and initials pattern features (5 features).
    
    Args:
        raw_str: Original raw string
        
    Returns:
        Array of 5 features
    """
    features = np.zeros(5, dtype=np.float32)
    
    if not raw_str:
        return features
    
    # All caps ratio
    alpha_chars = [c for c in raw_str if c.isalpha()]
    if alpha_chars:
        features[0] = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    
    # Capitalized token ratio
    tokens = raw_str.split()
    if tokens:
        cap_count = sum(1 for t in tokens if t and t[0].isupper())
        features[1] = cap_count / len(tokens)
    
    # Initials pattern (A B Smith, A. B. Smith)
    initials_pattern = r'\b[A-Z]\.?\s+[A-Z]\.?\s+'
    features[2] = 1.0 if re.search(initials_pattern, raw_str) else 0.0
    
    # Person title
    normalized = normalize(raw_str)
    if normalized.tokens:
        first_token = normalized.tokens[0]
        features[3] = 1.0 if first_token in PERSON_TITLES else 0.0
        
        # Person suffix (check last token)
        last_token = normalized.tokens[-1]
        features[4] = 1.0 if last_token in PERSON_SUFFIXES else 0.0
    
    return features


def extract_lexical_features(name: str) -> np.ndarray:
    """Extract lexical signal features (3 features).
    
    Args:
        name: Name to extract features from
        
    Returns:
        Array of 3 features
    """
    features = np.zeros(3, dtype=np.float32)
    
    normalized = normalize(name)
    tokens_set = set(normalized.tokens)
    
    # Organization keyword
    features[0] = 1.0 if any(kw in tokens_set for kw in ORG_KEYWORDS) else 0.0
    
    # Given name hit
    features[1] = 1.0 if any(name in tokens_set for name in GIVEN_NAMES) else 0.0
    
    # Surname particle
    features[2] = 1.0 if any(particle in tokens_set for particle in SURNAME_PARTICLES) else 0.0
    
    return features


def extract_features(
    name: str,
    iso_matcher,
    language_hint: str = None,
    jurisdiction_hint: str = None
) -> np.ndarray:
    """Extract all engineered features from a name.
    
    UPDATED: Now returns 24 features (removed 8 unused language/jurisdiction features)
    
    Features (24 total):
    - ISO 20275-derived: 8 features (indices 0-7)
    - String structure: 8 features (indices 8-15)  
    - Casing/initials: 5 features (indices 16-20)
    - Lexical signals: 3 features (indices 21-23)
    
    Args:
        name: Name string to extract features from
        iso_matcher: ISO20275Matcher instance for legal form detection
        language_hint: Optional language code (e.g., 'en', 'de') - CURRENTLY UNUSED
        jurisdiction_hint: Optional jurisdiction code - CURRENTLY UNUSED
        
    Returns:
        NumPy array of shape (24,) with float32 dtype
    """
    # Normalize the name
    normalized = normalize(name)
    
    # Extract feature groups
    iso_features = extract_iso_features(name, iso_matcher)  # 8 features
    # language_features removed (was 8, now 0)
    structure_features = extract_string_structure_features(normalized.norm_str, name)  # 8 features
    casing_features = extract_casing_initials_features(name)  # 5 features (pass raw name)
    lexical_features = extract_lexical_features(normalized)  # 3 features
    
    # Concatenate all features
    all_features = np.concatenate([
        iso_features,
        structure_features,
        casing_features,
        lexical_features
    ])
    
    return all_features


def get_feature_names() -> list:
    """Get the names of all engineered features in order.
    
    Returns:
        List of 24 feature names (reduced from 32 - removed unused language features)
    """
    return [
        # ISO 20275-derived (8)
        'has_legal_form',
        'legal_form_tier_a',
        'legal_form_tier_b',
        'legal_form_tier_c',
        'legal_form_token_len',
        'legal_form_char_len',
        'legal_form_is_ambiguous_short',
        'legal_form_country_is_known',
        
        # String structure (8)
        'char_len',
        'token_count',
        'avg_token_len',
        'contains_digit',
        'contains_ampersand',
        'contains_comma',
        'contains_slash_or_hyphen',
        'contains_period',
        
        # Casing/initials (5)
        'all_caps_ratio',
        'capitalised_token_ratio',
        'has_initials_pattern',
        'has_person_title',
        'has_person_suffix',
        
        # Lexical signals (3)
        'has_org_keyword',
        'has_given_name_hit',
        'has_surname_particle'
    ]
