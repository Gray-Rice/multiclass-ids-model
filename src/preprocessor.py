import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.config import DROP_COLUMNS, LABEL_MAPPING

class LabelCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, invalid_labels=None, label_mapping=None):
        self.invalid_labels = invalid_labels or ['Label']
        self.label_mapping = label_mapping or LABEL_MAPPING

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'Label' in df.columns:
            mask = df['Label'].isin(self.invalid_labels)
            df = df[~mask]
            for old, new in self.label_mapping.items():
                df['Label'] = df['Label'].replace(old, new)
        return df

class InfinityHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return df

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        existing_cols = [c for c in self.columns if c in X.columns]
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Columns to drop: {existing_cols}")
        logger.info(f"Remaining columns: {len(X.columns) - len(existing_cols)}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        existing_cols = [c for c in self.columns if c in df.columns]
        return df.drop(columns=existing_cols, errors='ignore')

class NumericOnlySelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_cols_ = None

    def fit(self, X, y=None):
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        available_cols = [c for c in self.numeric_cols_ if c in df.columns]
        return df[available_cols]

def get_cleaning_pipeline() -> Pipeline:
    pipeline = Pipeline([
        ('label_cleaner', LabelCleaner()),
        ('column_dropper', ColumnDropper(DROP_COLUMNS)),
        ('infinity_handler', InfinityHandler())
    ])
    return pipeline

def get_model_pipeline(classifier) -> Pipeline:
    pipeline = Pipeline([
        ('numeric_selector', NumericOnlySelector()),
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=False)),
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    return pipeline
