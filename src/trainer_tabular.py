import json
import logging
from pathlib import Path

import joblib
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

from src.config import PROCESSED_DIR, REPORTS_DIR, RANDOM_STATE, N_JOBS
from src.preprocess import build_train_val_test_matrices

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


def run_tabular_training(cleaned_parquet: Path = PROCESSED_DIR / "cleaned_data.parquet"):
    if not cleaned_parquet.exists():
        raise FileNotFoundError(f"Missing {cleaned_parquet}")

    df = pd.read_parquet(cleaned_parquet)

    pack = build_train_val_test_matrices(
        df=df,
        label_col="Label",
        drop_cols=["__day_tag", "__source_file"],
        corr_threshold=0.98,
        scale=False,
    )

    X_train, X_val, X_test = pack["X_train"], pack["X_val"], pack["X_test"]
    y_train, y_val, y_test = pack["y_train"], pack["y_val"], pack["y_test"]
    le = pack["label_encoder"]
    labels = list(range(len(le.classes_)))

    # Basic shape checks
    assert X_train.shape[0] == len(y_train), "X_train/y_train size mismatch"
    assert X_val.shape[0] == len(y_val), "X_val/y_val size mismatch"
    assert X_test.shape[0] == len(y_test), "X_test/y_test size mismatch"

    models = {
        "LightGBM": LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=8,
            min_child_samples=300,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
    }

    results = {
        "meta": {
            "split_meta": pack["split_meta"],
            "n_features": int(X_train.shape[1]),
            "classes": le.classes_.tolist(),
            "dropped_constant_cols": pack["dropped_constant_cols"],
            "dropped_corr_cols": pack["dropped_corr_cols"],
        },
        "models": {},
    }

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        results["models"][name] = {
            "validation": eval_metrics(y_val, val_pred, labels=labels),
            "test": eval_metrics(y_test, test_pred, labels=labels),
            "test_classification_report": classification_report(
                y_test,
                test_pred,
                labels=labels,                 # IMPORTANT FIX
                target_names=le.classes_,      # must align with labels
                output_dict=True,
                zero_division=0,
            ),
        }

        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame(
                {"feature": pack["feature_names"], "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            fi.to_csv(REPORTS_DIR / f"feature_importance_{name.lower()}.csv", index=False)

        joblib.dump(model, PROCESSED_DIR / f"{name.lower()}_model.pkl")

    joblib.dump(le, PROCESSED_DIR / "label_encoder.pkl")
    pd.Series(pack["feature_names"]).to_csv(PROCESSED_DIR / "feature_names.csv", index=False)

    with open(REPORTS_DIR / "tabular_results.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Saved results to {REPORTS_DIR / 'tabular_results.json'}")
    return results


if __name__ == "__main__":
    out = run_tabular_training()
    for m, payload in out["models"].items():
        print(
            f"{m} -> val f1_macro={payload['validation']['f1_macro']:.4f}, "
            f"test f1_macro={payload['test']['f1_macro']:.4f}"
        )