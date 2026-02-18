import pandas as pd
import numpy as np
import json
import logging
from src.config import REPORTS_DIR, CV_FOLDS
from src.data_loader import load_datasets, optimize_memory
from src.preprocessor import get_cleaning_pipeline
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def audit_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes raw data for quirks without applying cleaning.
    """
    report = {
        "shape": df.shape,
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "missing_values": df.isnull().sum().to_dict(),
        "infinity_counts": {},
        "class_distribution": {},
        "data_types": df.dtypes.astype(str).to_dict(),
        "potential_issues": []
    }

    # Check for Infinity in numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_df).sum()
    report["infinity_counts"] = inf_counts[inf_counts > 0].to_dict()

    if report["infinity_counts"]:
        report["potential_issues"].append("Infinity values detected in numeric columns.")

    # Check for Header Leaks (Label == 'Label')
    if 'Label' in df.columns:
        leaks = df[df['Label'] == 'Label'].shape[0]
        if leaks > 0:
            report["potential_issues"].append(f"Found {leaks} header leak rows in 'Label' column.")

        # Class Distribution (Raw)
        report["class_distribution"] = df['Label'].value_counts().to_dict()

        # Check CV Feasibility
        for cls, count in report["class_distribution"].items():
            if count < CV_FOLDS:
                report["potential_issues"].append(f"Class '{cls}' has {count} samples. Insufficient for {CV_FOLDS}-Fold CV.")

    return report

def validate_pipeline(df: pd.DataFrame) -> bool:
    """
    Tests the proposed cleaning pipeline on a sample to ensure no errors.
    """
    logger.info("Validating proposed cleaning pipeline...")
    try:
        pipeline = get_cleaning_pipeline()
        # Use a sample for speed
        sample = df.sample(n=1000, random_state=42)
        cleaned_sample = pipeline.fit_transform(sample)

        logger.info(f"Pipeline validation successful. Output shape: {cleaned_sample.shape}")

        # Check for remaining NaNs after pipeline (before imputation)
        remaining_nans = cleaned_sample.isnull().sum().sum()
        if remaining_nans > 0:
            logger.warning(f"Pipeline left {remaining_nans} NaN values. Imputation required in Stage 2.")

        return True
    except Exception as e:
        logger.error(f"Pipeline validation failed: {e}")
        return False

def run_stage1_audit():
    """
    Main execution function for Stage 1.
    """
    logger.info("=== Starting Stage 1: Data Quality Audit ===")

    # 1. Load Raw Data
    try:
        df = load_datasets([]) # Populate file list in config or pass explicitly
        # Note: For this script to run, ensure src/config.py FILES list is correct
        # If FILES is empty in config, load_datasets will raise ValueError
    except ValueError as e:
        logger.error(e)
        return

    # 2. Optimize Memory
    df = optimize_memory(df)

    # 3. Audit Raw Data
    audit_report = audit_raw_data(df)

    # 4. Validate Cleaning Pipeline
    pipeline_valid = validate_pipeline(df)
    audit_report["pipeline_validation"] = "Passed" if pipeline_valid else "Failed"

    # 5. Save Report
    report_path = REPORTS_DIR / "stage1_audit_report.json"
    with open(report_path, 'w') as f:
        # Convert numpy types to python types for JSON serialization
        json.dump(audit_report, f, indent=4, default=str)

    logger.info(f"Audit report saved to {report_path}")
    logger.info("=== Stage 1 Complete ===")

    return audit_report

if __name__ == "__main__":
    run_stage1_audit()
