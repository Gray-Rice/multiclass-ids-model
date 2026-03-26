import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DATA_DIR,
    FILES,
    MAX_ROWS,
    N_JOBS,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.preprocessor import get_cleaning_pipeline, get_model_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------
# Data loading / cleaning
# -----------------------------
def _extract_day_tag(file_name: str) -> str:
    """
    Example: 'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv' -> 'Friday-02-03-2018'
    """
    return file_name.split("_")[0]


def load_and_clean_data_with_day_tag() -> Tuple[List[str], int]:
    """
    Loads CSVs in FILES order, applies cleaning pipeline per chunk, saves cleaned parquet per source file.
    Adds '__day_tag' and '__source_file' columns for strict temporal splitting.
    """
    logger.info("=== Starting Safe Data Loading & Cleaning (with day tags) ===")

    pipeline = get_cleaning_pipeline()
    total_rows = 0
    saved_files: List[str] = []

    for idx, file_name in enumerate(FILES):
        if MAX_ROWS and total_rows >= MAX_ROWS:
            logger.info(f"Reached MAX_ROWS ({MAX_ROWS}). Stopping load.")
            break

        src_path = DATA_DIR / file_name
        dst_path = PROCESSED_DIR / f"cleaned_{idx}.parquet"
        day_tag = _extract_day_tag(file_name)

        if not src_path.exists():
            logger.warning(f"Source file not found: {src_path}")
            continue

        if dst_path.exists():
            logger.info(f"Cleaned file {idx} already exists. Skipping...")
            temp_df = pd.read_parquet(dst_path)
            total_rows += len(temp_df)
            del temp_df
            saved_files.append(str(dst_path))
            continue

        logger.info(f"Processing {file_name}...")
        try:
            chunks = []
            loaded_rows_for_file = 0

            for chunk in pd.read_csv(src_path, chunksize=500_000, low_memory=False):
                cleaned_chunk = pipeline.fit_transform(chunk)  # rule-based transforms only
                cleaned_chunk["__day_tag"] = day_tag
                cleaned_chunk["__source_file"] = file_name
                chunks.append(cleaned_chunk)

                loaded_rows_for_file += len(cleaned_chunk)

                if MAX_ROWS and (total_rows + loaded_rows_for_file) >= MAX_ROWS:
                    break

                del chunk
                gc.collect()

            if chunks:
                file_df = pd.concat(chunks, ignore_index=True)

                if MAX_ROWS and len(file_df) + total_rows > MAX_ROWS:
                    file_df = file_df.iloc[: MAX_ROWS - total_rows]

                file_df.to_parquet(dst_path, index=False)
                saved_files.append(str(dst_path))
                total_rows += len(file_df)

                logger.info(f"Saved {dst_path.name}: {len(file_df)} rows")

                del file_df
                del chunks
                gc.collect()

        except Exception as e:
            logger.error(f"Failed to process {file_name}: {e}")
            continue

    if not saved_files:
        raise ValueError("No data was successfully cleaned and saved.")

    logger.info(f"Total rows cleaned: {total_rows}")
    return saved_files, total_rows


def combine_cleaned_data(saved_files: List[str]) -> pd.DataFrame:
    logger.info("Combining cleaned files...")
    dfs = []
    for f in saved_files:
        df = pd.read_parquet(f)
        dfs.append(df)
        del df
        gc.collect()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined shape: {combined.shape}")
    return combined


# -----------------------------
# Split + utilities
# -----------------------------
def temporal_day_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict day-based split using day order from FILES.
    """
    if "__day_tag" not in df.columns:
        raise ValueError("Missing '__day_tag'. Rebuild cleaned data with day tags.")

    ordered_days = [_extract_day_tag(f) for f in FILES]
    present_days = [d for d in ordered_days if d in set(df["__day_tag"].unique())]

    if len(present_days) < 3:
        raise ValueError("Need at least 3 distinct days for train/val/test temporal split.")

    n = len(present_days)
    train_end = max(1, int(n * train_ratio))
    val_end = min(n - 1, train_end + max(1, int(n * val_ratio)))

    if val_end <= train_end:
        val_end = train_end + 1
    if val_end >= n:
        val_end = n - 1

    train_days = present_days[:train_end]
    val_days = present_days[train_end:val_end]
    test_days = present_days[val_end:]

    if len(test_days) == 0:
        test_days = [present_days[-1]]
        val_days = present_days[train_end:-1]

    train_df = df[df["__day_tag"].isin(train_days)].copy()
    val_df = df[df["__day_tag"].isin(val_days)].copy()
    test_df = df[df["__day_tag"].isin(test_days)].copy()

    logger.info(f"Train days: {train_days}")
    logger.info(f"Val days:   {val_days}")
    logger.info(f"Test days:  {test_days}")
    logger.info(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    return train_df, val_df, test_df


def drop_high_correlation_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.98,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    corr = X_train[numeric_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        logger.info(f"Dropping {len(to_drop)} highly correlated features (>{threshold}).")
    else:
        logger.info("No highly correlated features to drop.")

    return (
        X_train.drop(columns=to_drop, errors="ignore"),
        X_val.drop(columns=to_drop, errors="ignore"),
        X_test.drop(columns=to_drop, errors="ignore"),
        to_drop,
    )


def _evaluate_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {
        str(lbl): {
            "precision": float(pp),
            "recall": float(rr),
            "f1": float(ff),
            "support": int(ss),
        }
        for lbl, pp, rr, ff, ss in zip(labels, p, r, f1, s)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "recall_macro": float(recall_macro),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized_true": cm_norm.tolist(),
    }


# -----------------------------
# Train / evaluate
# -----------------------------
def train_and_evaluate_temporal(df: pd.DataFrame) -> Dict:
    logger.info("Preparing temporal leakage-safe training...")

    if "Label" not in df.columns:
        raise ValueError("Label column missing.")

    # Remove exact duplicate rows (quick leakage guard)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Dropped duplicates: {before - len(df)}")

    train_df, val_df, test_df = temporal_day_split(df)

    # Label encode using train labels only
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["Label"])
    y_val = le.transform(val_df["Label"])
    y_test = le.transform(test_df["Label"])

    feature_drop_cols = ["Label", "__day_tag", "__source_file"]
    X_train = train_df.drop(columns=feature_drop_cols, errors="ignore")
    X_val = val_df.drop(columns=feature_drop_cols, errors="ignore")
    X_test = test_df.drop(columns=feature_drop_cols, errors="ignore")

    # Align columns
    common_cols = sorted(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
    X_train = X_train[common_cols].copy()
    X_val = X_val[common_cols].copy()
    X_test = X_test[common_cols].copy()

    # Drop all-NaN or constant features based on train only
    nunique = X_train.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        logger.info(f"Dropping {len(const_cols)} constant features.")
        X_train = X_train.drop(columns=const_cols, errors="ignore")
        X_val = X_val.drop(columns=const_cols, errors="ignore")
        X_test = X_test.drop(columns=const_cols, errors="ignore")

    # Optional: correlation-based pruning from train only
    X_train, X_val, X_test, corr_dropped = drop_high_correlation_features(
        X_train, X_val, X_test, threshold=0.98
    )

    # Persist artifacts
    joblib.dump(le, PROCESSED_DIR / "label_encoder.pkl")
    pd.Series(X_train.columns).to_csv(PROCESSED_DIR / "feature_names.csv", index=False)
    logger.info("Saved label encoder and selected feature names.")

    models = {
        "LightGBM": LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=8,
            min_child_samples=300,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            class_weight="balanced",
            verbose=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            class_weight="balanced_subsample",
        ),
    }

    results = {
        "meta": {
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "num_features_final": int(X_train.shape[1]),
            "corr_dropped_features": corr_dropped,
            "classes": le.classes_.tolist(),
        },
        "models": {},
    }

    label_ids = list(range(len(le.classes_)))

    for model_name, clf in models.items():
        logger.info(f"Training {model_name}...")

        pipeline = get_model_pipeline(clf)

        # Early stopping only for LightGBM
        if model_name == "LightGBM":
            pipeline.fit(
                X_train,
                y_train,
                classifier__eval_set=[(X_val.select_dtypes(include=[np.number]), y_val)],
                classifier__eval_metric="multi_logloss",
                classifier__callbacks=[],
            )
        else:
            pipeline.fit(X_train, y_train)

        # Predictions
        val_pred = pipeline.predict(X_val)
        test_pred = pipeline.predict(X_test)

        val_metrics = _evaluate_split(y_val, val_pred, labels=label_ids)
        test_metrics = _evaluate_split(y_test, test_pred, labels=label_ids)

        # Human-readable class report on test
        test_report = classification_report(
            y_test,
            test_pred,
            target_names=le.classes_,
            output_dict=True,
            zero_division=0,
        )

        results["models"][model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "test_classification_report": test_report,
        }

        # Feature importance where available
        try:
            importances = pipeline.named_steps["classifier"].feature_importances_
            fi_df = pd.DataFrame(
                {"feature": X_train.columns, "importance": importances}
            ).sort_values(by="importance", ascending=False)
            fi_path = REPORTS_DIR / f"feature_importance_{model_name.lower()}.csv"
            fi_df.to_csv(fi_path, index=False)
            logger.info(f"Saved feature importance: {fi_path}")
        except Exception:
            logger.info(f"{model_name} does not expose feature_importances_ in this pipeline.")

        # Save model
        model_path = PROCESSED_DIR / f"{model_name.lower()}_pipeline.pkl"
        joblib.dump(pipeline, model_path)
        logger.info(f"Saved model pipeline: {model_path}")

        del pipeline
        del clf
        gc.collect()

    # Save summary JSON
    report_path = REPORTS_DIR / "stage2_temporal_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved temporal results: {report_path}")

    return results


if __name__ == "__main__":
    cleaned_path = PROCESSED_DIR / "cleaned_data.parquet"

    if cleaned_path.exists():
        logger.info("Loading existing cleaned data...")
        df = pd.read_parquet(cleaned_path)
        # Backward compatibility if old cleaned file has no day tags
        if "__day_tag" not in df.columns:
            logger.warning("Existing cleaned_data.parquet has no __day_tag. Rebuilding from raw...")
            saved_files, _ = load_and_clean_data_with_day_tag()
            df = combine_cleaned_data(saved_files)
            df.to_parquet(cleaned_path, index=False)
    else:
        saved_files, _ = load_and_clean_data_with_day_tag()
        df = combine_cleaned_data(saved_files)
        df.to_parquet(cleaned_path, index=False)
        logger.info(f"Saved combined data to {cleaned_path}")

    train_and_evaluate_temporal(df)