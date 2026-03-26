import json
import logging
from pathlib import Path

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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from src.config import PROCESSED_DIR, REPORTS_DIR
from src.preprocess import build_train_val_test_matrices

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_sequences(X, y, seq_len=20):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
    if not Xs:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,), dtype=int)
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def eval_metrics(y_true, y_pred, labels):
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            str(lbl): {
                "precision": float(pp),
                "recall": float(rr),
                "f1": float(ff),
                "support": int(ss),
            }
            for lbl, pp, rr, ff, ss in zip(labels, p, r, f1, s)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_normalized_true": confusion_matrix(
            y_true, y_pred, labels=labels, normalize="true"
        ).tolist(),
    }


def build_lstm(input_shape, n_classes):
    model = Sequential(
        [
            LSTM(64, input_shape=input_shape),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def run_lstm_training(
    cleaned_parquet: Path = PROCESSED_DIR / "cleaned_data.parquet",
    seq_len: int = 20,
    epochs: int = 15,
    batch_size: int = 1024,
):
    if not cleaned_parquet.exists():
        raise FileNotFoundError(f"Missing {cleaned_parquet}")

    df = pd.read_parquet(cleaned_parquet)

    pack = build_train_val_test_matrices(
        df=df,
        label_col="Label",
        drop_cols=["__day_tag", "__source_file"],
        corr_threshold=0.98,
        scale=True,
    )

    X_train, X_val, X_test = pack["X_train"], pack["X_val"], pack["X_test"]
    y_train, y_val, y_test = pack["y_train"], pack["y_val"], pack["y_test"]
    le = pack["label_encoder"]
    labels = list(range(len(le.classes_)))

    X_train_seq, y_train_seq = build_sequences(X_train, y_train, seq_len=seq_len)
    X_val_seq, y_val_seq = build_sequences(X_val, y_val, seq_len=seq_len)
    X_test_seq, y_test_seq = build_sequences(X_test, y_test, seq_len=seq_len)

    if min(len(X_train_seq), len(X_val_seq), len(X_test_seq)) == 0:
        raise ValueError("Not enough rows for sequence length. Reduce seq_len (e.g., 5 or 10).")

    n_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train_seq, num_classes=n_classes)
    y_val_cat = to_categorical(y_val_seq, num_classes=n_classes)

    model = build_lstm((X_train_seq.shape[1], X_train_seq.shape[2]), n_classes=n_classes)
    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    hist = model.fit(
        X_train_seq,
        y_train_cat,
        validation_data=(X_val_seq, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1,
    )

    val_pred = np.argmax(model.predict(X_val_seq, batch_size=batch_size, verbose=0), axis=1)
    test_pred = np.argmax(model.predict(X_test_seq, batch_size=batch_size, verbose=0), axis=1)

    results = {
        "meta": {
            "split_meta": pack["split_meta"],
            "n_features": int(X_train.shape[1]),
            "classes": le.classes_.tolist(),
            "sequence_length": int(seq_len),
            "train_sequences": int(len(X_train_seq)),
            "val_sequences": int(len(X_val_seq)),
            "test_sequences": int(len(X_test_seq)),
            "epochs_ran": int(len(hist.history.get("loss", []))),
        },
        "validation": eval_metrics(y_val_seq, val_pred, labels=labels),
        "test": eval_metrics(y_test_seq, test_pred, labels=labels),
        "test_classification_report": classification_report(
            y_test_seq,
            test_pred,
            labels=labels,                 # IMPORTANT FIX
            target_names=le.classes_,      # aligns with labels
            output_dict=True,
            zero_division=0,
        ),
    }

    model.save(PROCESSED_DIR / "lstm_model.keras")
    joblib.dump(pack["scaler"], PROCESSED_DIR / "lstm_scaler.pkl")
    joblib.dump(le, PROCESSED_DIR / "lstm_label_encoder.pkl")
    pd.Series(pack["feature_names"]).to_csv(PROCESSED_DIR / "lstm_feature_names.csv", index=False)

    with open(REPORTS_DIR / "lstm_results.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Saved LSTM results to {REPORTS_DIR / 'lstm_results.json'}")
    return results


if __name__ == "__main__":
    out = run_lstm_training(seq_len=20, epochs=15, batch_size=1024)
    print(
        f"LSTM -> val f1_macro={out['validation']['f1_macro']:.4f}, "
        f"test f1_macro={out['test']['f1_macro']:.4f}"
    )