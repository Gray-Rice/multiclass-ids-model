import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder

logger = logging.getLogger(__name__)


# -----------------------------
# Split helpers
# -----------------------------
def temporal_or_fallback_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    day_col: str = "__day_tag",
    timestamp_candidates: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Split priority:
      1) day_col if present
      2) timestamp column if present
      3) stable row-order split
    """
    if timestamp_candidates is None:
        timestamp_candidates = ["Timestamp", "timestamp", "time", "date", "datetime"]

    meta = {"split_mode": None, "details": {}}

    # 1) Day-tag split
    if day_col in df.columns:
        ordered_days = pd.Series(df[day_col].dropna().unique()).tolist()
        # Keep first occurrence order as found in data (assumes parquet preserves append/file order)
        n = len(ordered_days)
        if n >= 3:
            train_end = max(1, int(n * train_ratio))
            val_end = min(n - 1, train_end + max(1, int(n * val_ratio)))
            if val_end <= train_end:
                val_end = train_end + 1
            if val_end >= n:
                val_end = n - 1

            train_days = ordered_days[:train_end]
            val_days = ordered_days[train_end:val_end]
            test_days = ordered_days[val_end:]

            train_df = df[df[day_col].isin(train_days)].copy()
            val_df = df[df[day_col].isin(val_days)].copy()
            test_df = df[df[day_col].isin(test_days)].copy()

            meta["split_mode"] = "day_tag"
            meta["details"] = {
                "train_days": train_days,
                "val_days": val_days,
                "test_days": test_days,
            }
            return train_df, val_df, test_df, meta

    # 2) Timestamp split
    ts_col = None
    for c in timestamp_candidates:
        if c in df.columns:
            ts_col = c
            break

    if ts_col is not None:
        tmp = df.copy()
        tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors="coerce")
        tmp = tmp.sort_values(ts_col).reset_index(drop=True)

        n = len(tmp)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = tmp.iloc[:train_end].copy()
        val_df = tmp.iloc[train_end:val_end].copy()
        test_df = tmp.iloc[val_end:].copy()

        meta["split_mode"] = "timestamp"
        meta["details"] = {"timestamp_col": ts_col}
        return train_df, val_df, test_df, meta

    # 3) Row-order split fallback
    logger.warning("No __day_tag or timestamp found. Using row-order split.")
    tmp = df.reset_index(drop=True)

    n = len(tmp)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = tmp.iloc[:train_end].copy()
    val_df = tmp.iloc[train_end:val_end].copy()
    test_df = tmp.iloc[val_end:].copy()

    meta["split_mode"] = "row_order"
    meta["details"] = {}
    return train_df, val_df, test_df, meta


# -----------------------------
# Feature cleaning
# -----------------------------
def _safe_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.select_dtypes(include=[np.number]).copy()
    Xn.replace([np.inf, -np.inf], np.nan, inplace=True)
    return Xn


def _drop_constant_columns(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    nunique = X_train.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X_train = X_train.drop(columns=const_cols, errors="ignore")
        X_val = X_val.drop(columns=const_cols, errors="ignore")
        X_test = X_test.drop(columns=const_cols, errors="ignore")
    return X_train, X_val, X_test, const_cols


def _drop_high_corr_columns(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.98,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    if X_train.shape[1] == 0:
        return X_train, X_val, X_test, []

    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        X_train = X_train.drop(columns=to_drop, errors="ignore")
        X_val = X_val.drop(columns=to_drop, errors="ignore")
        X_test = X_test.drop(columns=to_drop, errors="ignore")

    return X_train, X_val, X_test, to_drop


def build_train_val_test_matrices(
    df: pd.DataFrame,
    label_col: str = "Label",
    drop_cols: Optional[List[str]] = None,
    corr_threshold: float = 0.98,
    scale: bool = True,
) -> Dict:
    """
    End-to-end quick preprocessing from a single dataframe.
    Returns leakage-safe matrices and fitted objects.
    """
    if drop_cols is None:
        drop_cols = ["__day_tag", "__source_file"]

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    # Drop exact duplicates first
    df = df.drop_duplicates().reset_index(drop=True)

    # Split
    train_df, val_df, test_df, split_meta = temporal_or_fallback_split(df)

    # Encode labels (fit train only)
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col])
    y_val = le.transform(val_df[label_col])
    y_test = le.transform(test_df[label_col])

    # Build feature frames
    hard_drop = set(drop_cols + [label_col])
    feature_cols = [c for c in train_df.columns if c not in hard_drop]
    # Align common columns
    common = sorted(list(set(feature_cols) & set(val_df.columns) & set(test_df.columns)))

    X_train = train_df[common].copy()
    X_val = val_df[common].copy()
    X_test = test_df[common].copy()

    # Numeric only (fast + stable for tabular/LSTM baseline)
    X_train = _safe_numeric_frame(X_train)
    X_val = _safe_numeric_frame(X_val[X_train.columns.intersection(X_val.columns)])
    X_test = _safe_numeric_frame(X_test[X_train.columns.intersection(X_test.columns)])

    # Re-align exactly
    common_num = sorted(list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns)))
    X_train = X_train[common_num].copy()
    X_val = X_val[common_num].copy()
    X_test = X_test[common_num].copy()

    # Impute with train medians only
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    # Drop constant + high correlation (fit decisions on train only)
    X_train, X_val, X_test, const_cols = _drop_constant_columns(X_train, X_val, X_test)
    X_train, X_val, X_test, corr_cols = _drop_high_corr_columns(
        X_train, X_val, X_test, threshold=corr_threshold
    )

    scaler = None
    if scale:
        scaler = RobustScaler()
        X_train_arr = scaler.fit_transform(X_train.values)
        X_val_arr = scaler.transform(X_val.values)
        X_test_arr = scaler.transform(X_test.values)
    else:
        X_train_arr = X_train.values
        X_val_arr = X_val.values
        X_test_arr = X_test.values

    return {
        "X_train": X_train_arr,
        "X_val": X_val_arr,
        "X_test": X_test_arr,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": X_train.columns.tolist(),
        "label_encoder": le,
        "scaler": scaler,
        "split_meta": split_meta,
        "dropped_constant_cols": const_cols,
        "dropped_corr_cols": corr_cols,
    }