import logging
import os
from src.config import PROCESSED_DIR, REPORTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    if not os.path.exists(REPORTS_DIR / "stage1_audit_report.json"):
        logger.info("Running Stage 1 Audit...")
        from src.explorer import run_stage1_audit
        run_stage1_audit()
    else:
        logger.info("Stage 1 Audit found. Skipping...")

    logger.info("Starting Stage 2: Training and Evaluation...")
    try:
        from src.trainer import load_and_clean_data_safe, combine_cleaned_data, train_and_evaluate

        cleaned_path = PROCESSED_DIR / "cleaned_data.parquet"
        if cleaned_path.exists():
            import pandas as pd
            df = pd.read_parquet(cleaned_path)
        else:
            saved_files, total_rows = load_and_clean_data_safe()
            df = combine_cleaned_data(saved_files)
            df.to_parquet(cleaned_path, index=False)
            logger.info(f"Saved combined data to {cleaned_path}")

        results = train_and_evaluate(df)

        print("\n--- Stage 2 Complete ---")
        for model, metrics in results.items():
            print(f"{model} Macro F1: {metrics['f1_macro']['mean']:.4f} (+/- {metrics['f1_macro']['std']:.4f})")

    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")
        raise

if __name__ == "__main__":
    main()
