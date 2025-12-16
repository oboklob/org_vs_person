"""Feature dropout utility for training robustness.

Implements tier-conditional dropout on ISO legal form features during training
to prevent over-reliance on legal form signals while retaining their value.
"""
import numpy as np
from typing import Optional


def apply_feature_dropout(
    X_features: np.ndarray,
    tier_metadata: list,
    tier_ab_rate: float = 0.3,
    tier_c_rate: float = 0.5,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Apply tier-conditional dropout to ISO legal form features.
    
    Randomly zeros out ISO-derived features (indices 0-7) based on legal form tier:
    - Tier A/B: dropout at tier_ab_rate (default 30%)
    - Tier C (ambiguous): dropout at tier_c_rate (default 50%)
    
    NOTE: Feature count reduced from 32 to 24 (removed 8 unused language features).
    ISO features are still indices 0-7.
    
    Args:
        X_features: Feature matrix of shape (n_samples, n_features)
                   First 8 features should be ISO-derived features
        tier_metadata: List of tier strings ('A', 'B', 'C', or None) for each sample
        tier_ab_rate: Dropout rate for Tier A/B forms (0.0 to 1.0)
        tier_c_rate: Dropout rate for Tier C (ambiguous) forms (0.0 to 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        Feature matrix with dropout applied (same shape as input)
        
    Example:
        >>> features = np.array([[1, 1, 0, 0, 2, 5, 1, 0, ...], ...])
        >>> tiers = ['A', 'C', None, 'B', ...]
        >>> X_dropout = apply_feature_dropout(features, tiers, 0.3, 0.5, 42)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Create a copy to avoid modifying original
    X_dropped = X_features.copy()
    
    # ISO features are indices 0-7
    iso_feature_indices = list(range(8))
    
    for i, tier in enumerate(tier_metadata):
        if tier is None:
            # No legal form, no dropout
            continue
        
        # Determine dropout rate based on tier
        if tier in ['A', 'B']:
            dropout_rate = tier_ab_rate
        elif tier == 'C':
            dropout_rate = tier_c_rate
        else:
            # Unknown tier, skip
            continue
        
        # Apply dropout with probability
        if np.random.random() < dropout_rate:
            # Zero out ISO features for this sample
            X_dropped[i, iso_feature_indices] = 0
    
    return X_dropped


def extract_tier_metadata(names: list, iso_matcher) -> list:
    """Extract tier metadata for each name.
    
    Args:
        names: List of names
        iso_matcher: ISO20275Matcher instance
        
    Returns:
        List of tier strings ('A', 'B', 'C', or None) for each name
    """
    tiers = []
    for name in names:
        match = iso_matcher.match_legal_form(name)
        if match:
            tiers.append(match.metadata.tier)
        else:
            tiers.append(None)
    return tiers
