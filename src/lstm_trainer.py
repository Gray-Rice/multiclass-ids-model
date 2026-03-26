import gc
import json
import logging
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from src.config import FILES, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _extract_day_tag(file_name: str) -> str:
    return file_name.split("_")[0]


def temporal_day_split(
    df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "__day_tag" not in df.columns:
        raise ValueError("Missing '__day_tag' in dataframe. Rebuild cleaned data with day tags.")

    ordered_days = [_extract_day_tag(f) for f in FILES]
    present_days = [d for d in ordered_days if d in set(df["__day_tag"].unique())]

    if len(present_days) < 3:
        raise ValueError("Need at least 3 distinct days for train/val/test split.")

    n = len(present_days)
    train_end = max(1, int(n * train_ratio))
    val_end = min(n - 1, train_end + max(1, int(n * val_ratio)))

    if val_end <= train_end:
        val_end = train_end + 1
    if val_end >= n:
        val_end = n - 1

    train_days = present_days[:train_end]
    val_days = present_days[train_end:val_end]
    test_days = present_days[val_end:] if present_days[val_end:] else [present_days[-1]]

    train_df = df[df["__day_tag"].isin(train_days)].copy()
    val_df = df[df["__day_tag"].isin(val_days)].copy()
    test_df = df[df["__day_tag"].isin(test_days)].copy()

    logger.info(f"[LSTM] Train days: {train_days}")
    logger.info(f"[LSTM] Val days:   {val_days}")
    logger.info(f"[LSTM] Test days:  {test_days}")
    logger.info(
        f"[LSTM] Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

    return train_df, val_df, test_df


def _prepare_xy(df: pd.DataFrame, feature_cols: List[str], label_col: str = "Label"):
    X = df[feature_cols].copy()
    y = df[label_col].copy()
    return X, y


def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sequence label = class of final element in the window.
    """
    if len(X) <= seq_len:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,), dtype=int)

    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len : i])
        y_seq.append(y[i])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.int64)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> Dict:
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


def _build_lstm_model(input_shape: Tuple[int, int], n_classes: int) -> Sequential:
    model = Sequential(
        [
            LSTM(64, input_shape=input_shape, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate_lstm(
    seq_len: int = 20,
    epochs: int = 20,
    batch_size: int = 1024,
) -> Dict:
    logger.info("=== LSTM temporal-safe training start ===")

    cleaned_path = PROCESSED_DIR / "cleaned_data.parquet"
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"{cleaned_path} not found. Run tabular pipeline first to generate cleaned data."
        )

    df = pd.read_parquet(cleaned_path)
    if "__day_tag" not in df.columns:
        raise ValueError("cleaned_data.parquet missing __day_tag. Rebuild with day-tag trainer loader.")

    if "Label" not in df.columns:
        raise ValueError("Label column missing.")

    # Deduplicate for leakage robustness
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"[LSTM] Dropped duplicates: {before - len(df)}")

    train_df, val_df, test_df = temporal_day_split(df)

    drop_cols = ["Label", "__day_tag", "__source_file"]
    feature_cols = sorted(
        list(
            (set(train_df.columns) & set(val_df.columns) & set(test_df.columns))
            - set(drop_cols)
        )
    )

    # Keep numeric only for LSTM baseline
    train_X_df = train_df[feature_cols].select_dtypes(include=[np.number]).copy()
    val_X_df = val_df[train_X_df.columns].copy()
    test_X_df = test_df[train_X_df.columns].copy()

    # Drop constant columns from train only
    nunique = train_X_df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        train_X_df = train_X_df.drop(columns=const_cols, errors="ignore")
        val_X_df = val_X_df.drop(columns=const_cols, errors="ignore")
        test_X_df = test_X_df.drop(columns=const_cols, errors="ignore")

    # Fill inf/nan safely
    for xdf in [train_X_df, val_X_df, test_X_df]:
        xdf.replace([np.inf, -np.inf], np.nan, inplace=True)

    medians = train_X_df.median()
    train_X_df = train_X_df.fillna(medians)
    val_X_df = val_X_df.fillna(medians)
    test_X_df = test_X_df.fillna(medians)

    # Label encoding (fit on train labels only)
    le = LabelEncoder()
    y_train_raw = le.fit_transform(train_df["Label"])
    y_val_raw = le.transform(val_df["Label"])
    y_test_raw = le.transform(test_df["Label"])
    n_classes = len(le.classes_)

    # Robust scaling (fit train only)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train_X_df.values)
    X_val_scaled = scaler.transform(val_X_df.values)
    X_test_scaled = scaler.transform(test_X_df.values)

    # Build fixed-length sequences
    X_train_seq, y_train_seq = _build_sequences(X_train_scaled, y_train_raw, seq_len=seq_len)
    X_val_seq, y_val_seq = _build_sequences(X_val_scaled, y_val_raw, seq_len=seq_len)
    X_test_seq, y_test_seq = _build_sequences(X_test_scaled, y_test_raw, seq_len=seq_len)

    if len(X_train_seq) == 0 or len(X_val_seq) == 0 or len(X_test_seq) == 0:
        raise ValueError("Not enough rows to build sequences. Reduce seq_len or use more data.")

    y_train_cat = to_categorical(y_train_seq, num_classes=n_classes)
    y_val_cat = to_categorical(y_val_seq, num_classes=n_classes)

    model = _build_lstm_model(
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        n_classes=n_classes,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train_seq,
        y_train_cat,
        validation_data=(X_val_seq, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    # Predict
    val_pred_proba = model.predict(X_val_seq, batch_size=batch_size, verbose=0)
    test_pred_proba = model.predict(X_test_seq, batch_size=batch_size, verbose=0)

    val_pred = np.argmax(val_pred_proba, axis=1)
    test_pred = np.argmax(test_pred_proba, axis=1)

    label_ids = list(range(n_classes))
    val_metrics = _evaluate(y_val_seq, val_pred, labels=label_ids)
    test_metrics = _evaluate(y_test_seq, test_pred, labels=label_ids)

    test_report = classification_report(
        y_test_seq,
        test_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "meta": {
            "sequence_length": int(seq_len),
            "num_features": int(X_train_seq.shape[2]),
            "num_classes": int(n_classes),
            "train_sequences": int(len(X_train_seq)),
            "val_sequences": int(len(X_val_seq)),
            "test_sequences": int(len(X_test_seq)),
            "classes": le.classes_.tolist(),
            "epochs_ran": int(len(history.history.get("loss", []))),
        },
        "validation": val_metrics,
        "test": test_metrics,
        "test_classification_report": test_report,
    }

    # Save artifacts
    model.save(PROCESSED_DIR / "lstm_model.keras")
    joblib.dump(scaler, PROCESSED_DIR / "lstm_scaler.pkl")
    joblib.dump(le, PROCESSED_DIR / "lstm_label_encoder.pkl")
    pd.Series(train_X_df.columns).to_csv(PROCESSED_DIR / "lstm_feature_names.csv", index=False)

    with open(REPORTS_DIR / "lstm_results.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Saved LSTM artifacts and reports.")
    logger.info(f"LSTM Val Macro F1:  {results['validation']['f1_macro']:.4f}")
    logger.info(f"LSTM Test Macro F1: {results['test']['f1_macro']:.4f}")

    gc.collect()
    return results


if __name__ == "__main__":
    train_and_evaluate_lstm()