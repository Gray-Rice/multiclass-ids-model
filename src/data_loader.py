import pandas as pd
import logging
from pathlib import Path
from typing import List
from src.config import DATA_DIR, DROP_COLUMNS

logger = logging.getLogger(__name__)

def load_datasets(file_names: List[str]) -> pd.DataFrame:
    """
    Loads and concatenates all CSV files from the raw data directory.
    Handles basic memory optimization by downcasting numeric types.
    """
    dataframes = []

    for file_name in file_names:
        file_path = DATA_DIR / file_name
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue

        logger.info(f"Loading {file_name}...")
        try:
            # low_memory=False prevents mixed type warnings
            df = pd.read_csv(file_path, low_memory=False)
            dataframes.append(df)
        except Exception as e:
            logger.error(f"Error loading {file_name}: {e}")

    if not dataframes:
        raise ValueError("No data loaded. Check file paths.")

    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined dataset shape: {combined_df.shape}")

    return combined_df

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcasts numeric columns to reduce memory footprint.
    """
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df
