import logging
from src.config import PROCESSED_DIR, REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    if not (REPORTS_DIR / "stage1_audit_report.json").exists():
        logger.info("Running Stage 1 Audit...")
        from src.explorer import run_stage1_audit

        run_stage1_audit()
    else:
        logger.info("Stage 1 Audit found. Skipping...")

    logger.info("Starting Stage 2: Temporal-safe Training and Evaluation...")
    try:
        from src.trainer import (
            load_and_clean_data_with_day_tag,
            combine_cleaned_data,
            train_and_evaluate_temporal,
        )

        cleaned_path = PROCESSED_DIR / "cleaned_data.parquet"

        if cleaned_path.exists():
            import pandas as pd

            logger.info("Loading existing cleaned_data.parquet...")
            df = pd.read_parquet(cleaned_path)

            # If old cleaned file has no day tags, rebuild from raw
            if "__day_tag" not in df.columns:
                logger.warning(
                    "Existing cleaned_data.parquet has no __day_tag. Rebuilding from raw..."
                )
                saved_files, _ = load_and_clean_data_with_day_tag()
                df = combine_cleaned_data(saved_files)
                df.to_parquet(cleaned_path, index=False)
                logger.info(f"Saved rebuilt cleaned data to {cleaned_path}")
        else:
            saved_files, _ = load_and_clean_data_with_day_tag()
            df = combine_cleaned_data(saved_files)
            df.to_parquet(cleaned_path, index=False)
            logger.info(f"Saved combined data to {cleaned_path}")

        tabular_results = train_and_evaluate_temporal(df)

        print("\n--- Stage 2A Complete: Tabular Models ---")
        for model_name, payload in tabular_results["models"].items():
            val_f1 = payload["validation"]["f1_macro"]
            test_f1 = payload["test"]["f1_macro"]
            print(f"{model_name} Macro F1 -> val: {val_f1:.4f} | test: {test_f1:.4f}")

    except Exception as e:
        logger.error(f"Stage 2A (tabular) failed: {e}")
        raise

    logger.info("Starting Stage 2B: LSTM Baseline...")
    try:
        from src.lstm_trainer import train_and_evaluate_lstm

        lstm_results = train_and_evaluate_lstm()

        print("\n--- Stage 2B Complete: LSTM ---")
        print(
            f"LSTM Macro F1 -> val: {lstm_results['validation']['f1_macro']:.4f} | "
            f"test: {lstm_results['test']['f1_macro']:.4f}"
        )

    except Exception as e:
        logger.error(f"Stage 2B (LSTM) failed: {e}")
        raise


if __name__ == "__main__":
    main()