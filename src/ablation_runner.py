import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

from src.config import PROCESSED_DIR, REPORTS_DIR, RANDOM_STATE, N_JOBS
from src.preprocess import build_train_val_test_matrices

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Edit this list as needed (ordered by suspicion)
SUSPECT_FEATURES = [
    "Init Fwd Win Byts",
    "Flow Duration",
    "Fwd Seg Size Min",
    "Flow IAT Min",
    "Fwd IAT Min",
    "Flow IAT Mean",
    "Fwd Pkts/s",
    "Fwd IAT Max",
]


def _score(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _fit_eval_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    models = {
        "LightGBM": LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=15,         # tighter than before
            max_depth=5,
            min_child_samples=1000,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=2.0,
            reg_lambda=10.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
    }

    out = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        out[name] = {
            "validation": _score(y_val, val_pred),
            "test": _score(y_test, test_pred),
        }
    return out


def run_ablation(
    cleaned_parquet: Path = PROCESSED_DIR / "cleaned_data.parquet",
    top_k_steps: List[int] = [0, 1, 3, 5, 8],
):
    if not cleaned_parquet.exists():
        raise FileNotFoundError(f"Missing {cleaned_parquet}")

    df = pd.read_parquet(cleaned_parquet)

    # Build once, no scaling for tree models
    pack = build_train_val_test_matrices(
        df=df,
        label_col="Label",
        drop_cols=["__day_tag", "__source_file"],
        corr_threshold=0.98,
        scale=False,
    )

    feature_names = list(pack["feature_names"])
    X_train_full = pack["X_train"]
    X_val_full = pack["X_val"]
    X_test_full = pack["X_test"]
    y_train = pack["y_train"]
    y_val = pack["y_val"]
    y_test = pack["y_test"]

    assert X_train_full.shape[1] == len(feature_names), "Feature name alignment mismatch"

    results = {
        "meta": {
            "split_meta": pack["split_meta"],
            "n_features_initial": int(len(feature_names)),
            "suspect_features": SUSPECT_FEATURES,
            "top_k_steps": top_k_steps,
        },
        "experiments": [],
    }

    for k in top_k_steps:
        drop_set = set(SUSPECT_FEATURES[:k])
        keep_idx = [i for i, f in enumerate(feature_names) if f not in drop_set]
        kept_features = [feature_names[i] for i in keep_idx]
        dropped_present = [f for f in SUSPECT_FEATURES[:k] if f in feature_names]

        if len(keep_idx) == 0:
            logger.warning(f"Skipping k={k}: no features left after ablation.")
            continue

        logger.info(
            f"Ablation k={k}: dropping {len(dropped_present)} suspect features, "
            f"remaining={len(kept_features)}"
        )

        X_train = X_train_full[:, keep_idx]
        X_val = X_val_full[:, keep_idx]
        X_test = X_test_full[:, keep_idx]

        model_scores = _fit_eval_models(X_train, y_train, X_val, y_val, X_test, y_test)

        results["experiments"].append(
            {
                "k": int(k),
                "dropped_features": dropped_present,
                "n_features_remaining": int(len(kept_features)),
                "scores": model_scores,
            }
        )

    # Save JSON
    out_json = REPORTS_DIR / "ablation_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)

    # Save flat CSV for easy viewing
    rows = []
    for exp in results["experiments"]:
        k = exp["k"]
        n_feat = exp["n_features_remaining"]
        dropped = ",".join(exp["dropped_features"])
        for model_name, sc in exp["scores"].items():
            rows.append(
                {
                    "k": k,
                    "model": model_name,
                    "n_features_remaining": n_feat,
                    "dropped_features": dropped,
                    "val_f1_macro": sc["validation"]["f1_macro"],
                    "test_f1_macro": sc["test"]["f1_macro"],
                    "val_accuracy": sc["validation"]["accuracy"],
                    "test_accuracy": sc["test"]["accuracy"],
                }
            )

    out_csv = REPORTS_DIR / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    logger.info(f"Saved ablation JSON -> {out_json}")
    logger.info(f"Saved ablation CSV  -> {out_csv}")

    # Print compact table
    print("\n=== Ablation Summary ===")
    print(
        f"{'k':<4} {'Model':<14} {'Val F1(macro)':>15} {'Test F1(macro)':>16} "
        f"{'Val Acc':>10} {'Test Acc':>10} {'#Feat':>8}"
    )
    print("-" * 88)
    for r in rows:
        print(
            f"{r['k']:<4} {r['model']:<14} "
            f"{r['val_f1_macro']:>15.4f} {r['test_f1_macro']:>16.4f} "
            f"{r['val_accuracy']:>10.4f} {r['test_accuracy']:>10.4f} "
            f"{r['n_features_remaining']:>8}"
        )

    return results


if __name__ == "__main__":
    run_ablation()