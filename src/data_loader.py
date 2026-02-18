import pandas as pd
import logging
import gc
from pathlib import Path
from typing import List, Optional
from src.config import DATA_DIR, FILES, DROP_COLUMNS

logger = logging.getLogger(__name__)

def load_and_optimize_file(file_name: str) -> Optional[pd.DataFrame]:
    """
    Loads a single CSV, optimizes memory, and returns it.
    Caller is responsible for deleting the DataFrame to free memory.
    """
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return None

    logger.info(f"Loading {file_name}...")
    try:
        # Load file
        df = pd.read_csv(file_path, low_memory=False)

        # Immediate Optimization
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif col_type == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')

        logger.info(f"Loaded {file_name}: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_name}: {e}")
        return None

def get_file_iterator(file_names: List[str]):
    """
    Generator that yields processed DataFrames one by one to save memory.
    """
    for file_name in file_names:
        df = load_and_optimize_file(file_name)
        if df is not None:
            yield df
            # Force garbage collection after yielding to free memory
            del df
            gc.collect()
