import json
import logging
from pathlib import Path

from src.config import REPORTS_DIR
from src.trainer_tabular import run_tabular_training
from src.trainer_lstm import run_lstm_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _safe_get(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _print_comparison(tabular_results: dict, lstm_results: dict):
    rows = []

    # Tabular models
    for model_name, payload in tabular_results.get("models", {}).items():
        rows.append(
            {
                "model": model_name,
                "val_f1_macro": _safe_get(payload, "validation", "f1_macro", default=float("nan")),
                "test_f1_macro": _safe_get(payload, "test", "f1_macro", default=float("nan")),
                "val_accuracy": _safe_get(payload, "validation", "accuracy", default=float("nan")),
                "test_accuracy": _safe_get(payload, "test", "accuracy", default=float("nan")),
            }
        )

    # LSTM model
    rows.append(
        {
            "model": "LSTM",
            "val_f1_macro": _safe_get(lstm_results, "validation", "f1_macro", default=float("nan")),
            "test_f1_macro": _safe_get(lstm_results, "test", "f1_macro", default=float("nan")),
            "val_accuracy": _safe_get(lstm_results, "validation", "accuracy", default=float("nan")),
            "test_accuracy": _safe_get(lstm_results, "test", "accuracy", default=float("nan")),
        }
    )

    # Sort by test macro F1 descending
    rows = sorted(rows, key=lambda x: x["test_f1_macro"], reverse=True)

    print("\n=== Model Comparison (Validation/Test) ===")
    print(f"{'Model':<15} {'Val F1(macro)':>15} {'Test F1(macro)':>16} {'Val Acc':>12} {'Test Acc':>12}")
    print("-" * 74)
    for r in rows:
        print(
            f"{r['model']:<15} "
            f"{r['val_f1_macro']:>15.4f} "
            f"{r['test_f1_macro']:>16.4f} "
            f"{r['val_accuracy']:>12.4f} "
            f"{r['test_accuracy']:>12.4f}"
        )


def main():
    logger.info("Running tabular training...")
    tabular_results = run_tabular_training()

    logger.info("Running LSTM training...")
    lstm_results = run_lstm_training(seq_len=20, epochs=15, batch_size=1024)

    # Save combined summary
    combined = {
        "tabular_results_file": str(REPORTS_DIR / "tabular_results.json"),
        "lstm_results_file": str(REPORTS_DIR / "lstm_results.json"),
        "tabular": tabular_results,
        "lstm": lstm_results,
    }

    combined_path = REPORTS_DIR / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=4)

    logger.info(f"Saved combined results to {combined_path}")

    _print_comparison(tabular_results, lstm_results)


if __name__ == "__main__":
    main()