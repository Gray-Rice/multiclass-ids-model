import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.config import DROP_COLUMNS

class LabelCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean specific label quirks in CIC-IDS2018.
    """
    def __init__(self, invalid_labels=None, label_mapping=None):
        self.invalid_labels = invalid_labels or ['Label']
        self.label_mapping = label_mapping or {'Infilteration': 'Infiltration'}

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'Label' in df.columns:
            # Drop rows where Label is invalid (header leaks)
            mask = df['Label'].isin(self.invalid_labels)
            df = df[~mask]

            # Apply label corrections
            for old, new in self.label_mapping.items():
                df['Label'] = df['Label'].replace(old, new)
        return df

class InfinityHandler(BaseEstimator, TransformerMixin):
    """
    Replaces infinity values with NaN so imputers can handle them.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return df

class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drops specified columns.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        existing_cols = [c for c in self.columns if c in df.columns]
        return df.drop(columns=existing_cols, errors='ignore')

def get_cleaning_pipeline() -> Pipeline:
    """
    Constructs a sklearn Pipeline for data cleaning.
    """
    pipeline = Pipeline([
        ('label_cleaner', LabelCleaner()),
        ('column_dropper', ColumnDropper(DROP_COLUMNS)),
        ('infinity_handler', InfinityHandler())
    ])
    return pipeline
