import json
import logging
from pathlib import Path
from typing import Dict, List

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
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from src.config import PROCESSED_DIR, REPORTS_DIR
from src.preprocess import build_train_val_test_matrices

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 20):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
    if not Xs:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,), dtype=int)
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> Dict:
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


def build_bigru(input_shape, n_classes: int) -> Sequential:
    model = Sequential(
        [
            Bidirectional(
                GRU(
                    64,
                    return_sequences=True,
                    dropout=0.2,              # input dropout
                    recurrent_dropout=0.2,    # recurrent dropout
                    kernel_regularizer=l2(1e-4),
                    recurrent_regularizer=l2(1e-4),
                ),
                input_shape=input_shape,
            ),
            Dropout(0.3),
            Bidirectional(
                GRU(
                    32,
                    return_sequences=False,
                    dropout=0.2,
                    recurrent_dropout=0.2,
                    kernel_regularizer=l2(1e-4),
                    recurrent_regularizer=l2(1e-4),
                )
            ),
            Dropout(0.3),
            Dense(64, activation="relu", kernel_regularizer=l2(1e-4)),
            Dropout(0.3),
            Dense(n_classes, activation="softmax"),
        ]
    )

    # gradient clipping is another overfitting/stability control
    optimizer = Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def save_confusion_csvs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    class_names: List[str],
    prefix: str = "bigru_test",
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)

    cm_path = REPORTS_DIR / f"{prefix}_confusion_matrix.csv"
    cm_norm_path = REPORTS_DIR / f"{prefix}_confusion_matrix_normalized.csv"

    cm_df.to_csv(cm_path, index=True)
    cm_norm_df.to_csv(cm_norm_path, index=True)

    logger.info(f"Saved confusion matrix CSV: {cm_path}")
    logger.info(f"Saved normalized confusion matrix CSV: {cm_norm_path}")


def run_bigru_training(
    cleaned_parquet: Path = PROCESSED_DIR / "cleaned_data.parquet",
    seq_len: int = 20,
    epochs: int = 30,
    batch_size: int = 1024,
):
    if not cleaned_parquet.exists():
        raise FileNotFoundError(f"Missing {cleaned_parquet}")

    df = pd.read_parquet(cleaned_parquet)

    # keep same split/preprocess style as LSTM trainer
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
    class_names = le.classes_.tolist()

    X_train_seq, y_train_seq = build_sequences(X_train, y_train, seq_len=seq_len)
    X_val_seq, y_val_seq = build_sequences(X_val, y_val, seq_len=seq_len)
    X_test_seq, y_test_seq = build_sequences(X_test, y_test, seq_len=seq_len)

    if min(len(X_train_seq), len(X_val_seq), len(X_test_seq)) == 0:
        raise ValueError("Not enough rows for sequence length. Reduce seq_len (e.g., 5 or 10).")

    n_classes = len(class_names)
    y_train_cat = to_categorical(y_train_seq, num_classes=n_classes)
    y_val_cat = to_categorical(y_val_seq, num_classes=n_classes)

    # class weights (helps imbalance + can reduce overfitting to majority classes)
    cls_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=y_train_seq,
    )
    class_weight = {i: float(w) for i, w in enumerate(cls_weights_arr)}

    model = build_bigru((X_train_seq.shape[1], X_train_seq.shape[2]), n_classes=n_classes)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            min_delta=1e-4,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train_seq,
        y_train_cat,
        validation_data=(X_val_seq, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    val_pred = np.argmax(model.predict(X_val_seq, batch_size=batch_size, verbose=0), axis=1)
    test_pred = np.argmax(model.predict(X_test_seq, batch_size=batch_size, verbose=0), axis=1)

    results = {
        "meta": {
            "split_meta": pack["split_meta"],
            "n_features": int(X_train.shape[1]),
            "classes": class_names,
            "sequence_length": int(seq_len),
            "train_sequences": int(len(X_train_seq)),
            "val_sequences": int(len(X_val_seq)),
            "test_sequences": int(len(X_test_seq)),
            "epochs_ran": int(len(history.history.get("loss", []))),
            "model": "BiGRU",
            "overfit_controls": {
                "gru_dropout": 0.2,
                "gru_recurrent_dropout": 0.2,
                "dense_dropout": 0.3,
                "l2": 1e-4,
                "early_stopping_patience": 5,
                "reduce_lr_on_plateau": True,
                "class_weight_balanced": True,
                "clipnorm": 1.0,
            },
        },
        "validation": eval_metrics(y_val_seq, val_pred, labels=labels),
        "test": eval_metrics(y_test_seq, test_pred, labels=labels),
        "test_classification_report": classification_report(
            y_test_seq,
            test_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    # Save model artifacts
    model.save(PROCESSED_DIR / "bigru_model.keras")
    joblib.dump(pack["scaler"], PROCESSED_DIR / "bigru_scaler.pkl")
    joblib.dump(le, PROCESSED_DIR / "bigru_label_encoder.pkl")
    pd.Series(pack["feature_names"]).to_csv(PROCESSED_DIR / "bigru_feature_names.csv", index=False)

    # Save JSON
    json_path = REPORTS_DIR / "bigru_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save compact metrics CSV
    metrics_rows = [
        {
            "split": "validation",
            "accuracy": results["validation"]["accuracy"],
            "f1_macro": results["validation"]["f1_macro"],
            "f1_weighted": results["validation"]["f1_weighted"],
            "recall_macro": results["validation"]["recall_macro"],
        },
        {
            "split": "test",
            "accuracy": results["test"]["accuracy"],
            "f1_macro": results["test"]["f1_macro"],
            "f1_weighted": results["test"]["f1_weighted"],
            "recall_macro": results["test"]["recall_macro"],
        },
    ]
    pd.DataFrame(metrics_rows).to_csv(REPORTS_DIR / "bigru_metrics_summary.csv", index=False)

    # Save full class report CSV
    pd.DataFrame(results["test_classification_report"]).transpose().to_csv(
        REPORTS_DIR / "bigru_test_classification_report.csv", index=True
    )

    # Save confusion matrices CSV
    save_confusion_csvs(y_test_seq, test_pred, labels, class_names, prefix="bigru_test")

    # Save predictions CSV
    pred_df = pd.DataFrame(
        {
            "y_true_id": y_test_seq,
            "y_pred_id": test_pred,
            "y_true_label": [class_names[i] for i in y_test_seq],
            "y_pred_label": [class_names[i] for i in test_pred],
        }
    )
    pred_df.to_csv(REPORTS_DIR / "bigru_test_predictions.csv", index=False)

    # Save training history CSV
    pd.DataFrame(history.history).to_csv(REPORTS_DIR / "bigru_training_history.csv", index=False)

    logger.info(f"Saved BiGRU results JSON: {json_path}")
    logger.info(f"BiGRU val f1_macro={results['validation']['f1_macro']:.4f}")
    logger.info(f"BiGRU test f1_macro={results['test']['f1_macro']:.4f}")

    return results


if __name__ == "__main__":
    out = run_bigru_training(seq_len=20, epochs=30, batch_size=1024)
    print(
        f"BiGRU -> val f1_macro={out['validation']['f1_macro']:.4f}, "
        f"test f1_macro={out['test']['f1_macro']:.4f}"
    )