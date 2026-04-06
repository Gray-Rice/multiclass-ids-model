import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import REPORTS_DIR


# -----------------------------
# Paths (edit if needed)
# -----------------------------
COMBINED_PATH = REPORTS_DIR / "combined_results.json"
BIGRU_PATH = REPORTS_DIR / "bigru_results.json"   # NOTE: file name expected from trainer_bigru.py
OUT_COMBINED_PATH = REPORTS_DIR / "combined_results.json"  # overwrite with merged version
OUT_CSV_PATH = REPORTS_DIR / "model_comparison_summary.csv"
PLOTS_DIR = REPORTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
    print(f"[OK] Saved JSON: {path}")


def safe_get(d: Dict[str, Any], *keys, default=np.nan):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def extract_rows(combined: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []

    # Tabular models
    tab_models = safe_get(combined, "tabular", "models", default={})
    if isinstance(tab_models, dict):
        for model_name, payload in tab_models.items():
            rows.append(
                {
                    "model": model_name,
                    "group": "tabular",
                    "val_f1_macro": safe_get(payload, "validation", "f1_macro"),
                    "test_f1_macro": safe_get(payload, "test", "f1_macro"),
                    "val_accuracy": safe_get(payload, "validation", "accuracy"),
                    "test_accuracy": safe_get(payload, "test", "accuracy"),
                }
            )

    # LSTM
    if "lstm" in combined and isinstance(combined["lstm"], dict):
        payload = combined["lstm"]
        rows.append(
            {
                "model": "LSTM",
                "group": "sequence",
                "val_f1_macro": safe_get(payload, "validation", "f1_macro"),
                "test_f1_macro": safe_get(payload, "test", "f1_macro"),
                "val_accuracy": safe_get(payload, "validation", "accuracy"),
                "test_accuracy": safe_get(payload, "test", "accuracy"),
            }
        )

    # BiGRU
    if "bigru" in combined and isinstance(combined["bigru"], dict):
        payload = combined["bigru"]
        rows.append(
            {
                "model": "BiGRU",
                "group": "sequence",
                "val_f1_macro": safe_get(payload, "validation", "f1_macro"),
                "test_f1_macro": safe_get(payload, "test", "f1_macro"),
                "val_accuracy": safe_get(payload, "validation", "accuracy"),
                "test_accuracy": safe_get(payload, "test", "accuracy"),
            }
        )

    return rows


def _annotate_bars(ax, bars, fmt="{:.4f}", y_offset=0.01, fontsize=9):
    """
    Add value labels on top of bars.
    """
    for b in bars:
        h = b.get_height()
        if pd.isna(h):
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + y_offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=0,
        )


def plot_metric_bars(df: pd.DataFrame):
    if df.empty:
        print("[WARN] Empty comparison dataframe, skipping metric plots.")
        return

    dfx = df.sort_values("test_f1_macro", ascending=False).reset_index(drop=True)

    x = np.arange(len(dfx))
    w = 0.35

    # -------------------------
    # Test metrics plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - w / 2, dfx["test_f1_macro"], width=w, label="Test Macro-F1")
    bars2 = ax.bar(x + w / 2, dfx["test_accuracy"], width=w, label="Test Accuracy")

    ax.set_xticks(x)
    ax.set_xticklabels(dfx["model"], rotation=20)
    ax.set_ylim(0, 1.08)  # extra headroom for labels
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Test)")
    ax.legend()

    _annotate_bars(ax, bars1, fmt="{:.4f}", y_offset=0.008, fontsize=9)
    _annotate_bars(ax, bars2, fmt="{:.4f}", y_offset=0.008, fontsize=9)

    fig.tight_layout()
    out = PLOTS_DIR / "model_comparison_test.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved plot: {out}")

    # -------------------------
    # Validation metrics plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - w / 2, dfx["val_f1_macro"], width=w, label="Val Macro-F1")
    bars2 = ax.bar(x + w / 2, dfx["val_accuracy"], width=w, label="Val Accuracy")

    ax.set_xticks(x)
    ax.set_xticklabels(dfx["model"], rotation=20)
    ax.set_ylim(0, 1.08)  # extra headroom for labels
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Validation)")
    ax.legend()

    _annotate_bars(ax, bars1, fmt="{:.4f}", y_offset=0.008, fontsize=9)
    _annotate_bars(ax, bars2, fmt="{:.4f}", y_offset=0.008, fontsize=9)

    fig.tight_layout()
    out = PLOTS_DIR / "model_comparison_validation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved plot: {out}")


def _confusion_for_model(combined: Dict[str, Any], model_name: str):
    """
    Returns (cm, class_names) for requested model on TEST split.
    """
    # Tabular model?
    tab_models = safe_get(combined, "tabular", "models", default={})
    tab_meta_classes = safe_get(combined, "tabular", "meta", "classes", default=[])
    if isinstance(tab_models, dict) and model_name in tab_models:
        cm = safe_get(tab_models[model_name], "test", "confusion_matrix", default=None)
        return cm, tab_meta_classes

    # LSTM?
    if model_name == "LSTM" and "lstm" in combined:
        cm = safe_get(combined["lstm"], "test", "confusion_matrix", default=None)
        classes = safe_get(combined["lstm"], "meta", "classes", default=tab_meta_classes)
        return cm, classes

    # BiGRU?
    if model_name == "BiGRU" and "bigru" in combined:
        cm = safe_get(combined["bigru"], "test", "confusion_matrix", default=None)
        classes = safe_get(combined["bigru"], "meta", "classes", default=tab_meta_classes)
        return cm, classes

    return None, []


def plot_confusion_matrices(combined: Dict[str, Any], model_names: List[str]):
    for model_name in model_names:
        cm, classes = _confusion_for_model(combined, model_name)
        if cm is None:
            print(f"[WARN] No confusion matrix found for {model_name}, skipping.")
            continue

        cm = np.array(cm, dtype=float)
        # Use shorter labels if very long
        labels = [str(c)[:22] + ("..." if len(str(c)) > 22 else "") for c in classes]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True if cm.size <= 100 else False,
            fmt=".0f",
            cmap="Blues",
            xticklabels=labels if len(labels) == cm.shape[1] else "auto",
            yticklabels=labels if len(labels) == cm.shape[0] else "auto",
            cbar=True,
        )
        plt.title(f"{model_name} - Test Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        out = PLOTS_DIR / f"confusion_matrix_{model_name.lower()}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[OK] Saved plot: {out}")


def merge_bigru_into_combined(combined: Dict[str, Any], bigru: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add/update combined['bigru'] and a file pointer.
    """
    combined["bigru_results_file"] = str(BIGRU_PATH)
    combined["bigru"] = bigru
    return combined


def main():
    print("=== Visualize + Merge Results ===")

    combined = load_json(COMBINED_PATH)
    if combined is None:
        raise FileNotFoundError(f"Cannot proceed without {COMBINED_PATH}")

    bigru = load_json(BIGRU_PATH)
    if bigru is not None:
        combined = merge_bigru_into_combined(combined, bigru)
        save_json(combined, OUT_COMBINED_PATH)
    else:
        print("[INFO] bigru_results.json not found; proceeding with existing combined results only.")

    # Build comparison table
    rows = extract_rows(combined)
    df = pd.DataFrame(rows)

    if df.empty:
        print("[WARN] No model rows extracted. Check combined_results.json schema.")
        return

    df = df.sort_values("test_f1_macro", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_CSV_PATH, index=False)
    print(f"[OK] Saved CSV: {OUT_CSV_PATH}")
    print("\n=== Model Summary ===")
    print(df.to_string(index=False))

    # Plots
    plot_metric_bars(df)

    model_names = df["model"].tolist()
    plot_confusion_matrices(combined, model_names)

    print(f"\nAll plots saved under: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
