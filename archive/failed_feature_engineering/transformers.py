"""Custom transformers for feature engineering pipeline."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from name_classifier.feature_engineering import extract_features, get_feature_names
from name_classifier.iso20275_matcher import ISO20275Matcher


class NameFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer for extracting engineered features from names.
    
    This transformer extracts 24 engineered features per name (reduced from 32)
    for use in a FeatureUnion pipeline alongside character n-grams.
    """
    
    def __init__(self, language_column=None, jurisdiction_column=None):
        """Initialize the feature extractor.
        
        Args:
            language_column: Optional column name for language hints in input DataFrame
            jurisdiction_column: Optional column name for jurisdiction hints
        """
        self.language_column = language_column
        self.jurisdiction_column = jurisdiction_column
        self.iso_matcher = None
        
    def fit(self, X, y=None):
        """Fit the transformer (loads ISO matcher).
        
        Args:
            X: Input data (array-like of names or DataFrame)
            y: Target values (ignored)
            
        Returns:
            self
        """
        # Initialize ISO matcher on first fit
        if self.iso_matcher is None:
            self.iso_matcher = ISO20275Matcher()
        return self
    
    def transform(self, X):
        """Transform names to feature matrix.
        
        Args:
            X: Input data (array-like of names or DataFrame)
            
        Returns:
            Feature matrix of shape (n_samples, 32)
        """
        if self.iso_matcher is None:
            raise RuntimeError("Transformer must be fit before transform")
        
        # Handle both array-like and DataFrame inputs
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            names = X['label'].values if 'label' in X.columns else X.iloc[:, 0].values
            language_hints = X[self.language_column].values if self.language_column and self.language_column in X.columns else None
            jurisdiction_hints = X[self.jurisdiction_column].values if self.jurisdiction_column and self.jurisdiction_column in X.columns else None
        else:
            names = X
            language_hints = None
            jurisdiction_hints = None
        
        # Extract features for each name
        features_list = []
        for i, name in enumerate(names):
            lang_hint = language_hints[i] if language_hints is not None else None
            juris_hint = jurisdiction_hints[i] if jurisdiction_hints is not None else None
            
            features = extract_features(
                name,
                self.iso_matcher,
                language_hint=lang_hint,
                jurisdiction_hint=juris_hint
            )
            features_list.append(features)
        
        return np.array(features_list, dtype=np.float32)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names for output.
        
        Returns:
            Array of feature names
        """
        return np.array(get_feature_names())


class TextExtractor(BaseEstimator, TransformerMixin):
    """Extract text column from DataFrame for use with HashingVectorizer.
    
    This is a simple helper to extract the 'label' column from a DataFrame
    so it can be passed to HashingVectorizer in a FeatureUnion.
    """
    
    def __init__(self, column='label'):
        """Initialize text extractor.
        
        Args:
            column: Column name to extract (default: 'label')
        """
        self.column = column
    
    def fit(self, X, y=None):
        """Fit (does nothing).
        
        Args:
            X: Input data
            y: Target values (ignored)
            
        Returns:
            self
        """
        return self
    
    def transform(self, X):
        """Extract text column.
        
        Args:
            X: Input DataFrame or array-like
            
        Returns:
            Array of text strings
        """
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X[self.column].values if self.column in X.columns else X.iloc[:, 0].values
        else:
            return X
