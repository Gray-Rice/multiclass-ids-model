import pandas as pd
import numpy as np
import json
import logging
import gc
from src.config import REPORTS_DIR, CV_FOLDS, FILES
from src.data_loader import get_file_iterator
from src.preprocessor import get_cleaning_pipeline
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def audit_iterative() -> Dict[str, Any]:
    """
    Audits data file-by-file to prevent OOM errors.
    """
    logger.info("=== Starting Iterative Data Audit ===")

    # Initialize aggregators
    total_rows = 0
    total_memory_mb = 0
    global_missing_values = {}
    global_infinity_counts = {}
    global_class_distribution = {}
    potential_issues = []
    files_processed = 0

    # Iterate through files without holding all in memory
    for df in get_file_iterator(FILES):
        files_processed += 1
        total_rows += len(df)
        total_memory_mb += df.memory_usage(deep=False).sum() / 1024**2

        # 1. Check Missing Values
        missing = df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                global_missing_values[col] = global_missing_values.get(col, 0) + count

        # 2. Check Infinity
        numeric_df = df.select_dtypes(include=[np.number])
        inf_counts = np.isinf(numeric_df).sum()
        for col, count in inf_counts.items():
            if count > 0:
                global_infinity_counts[col] = global_infinity_counts.get(col, 0) + count

        # 3. Check Class Distribution
        if 'Label' in df.columns:
            # Count before cleaning to detect leaks
            leaks = df[df['Label'] == 'Label'].shape[0]
            if leaks > 0:
                # We track this but don't add to issue list yet until end
                pass

            # Clean temporarily for distribution count
            temp_df = df[df['Label'] != 'Label']
            temp_df = temp_df.copy() # Avoid SettingWithCopyWarning
            if 'Infilteration' in temp_df['Label'].values:
                temp_df['Label'] = temp_df['Label'].replace('Infilteration', 'Infiltration')

            class_counts = temp_df['Label'].value_counts().to_dict()
            for cls, count in class_counts.items():
                global_class_distribution[cls] = global_class_distribution.get(cls, 0) + count

        # Clean up
        del df
        gc.collect()
        logger.info(f"Processed {files_processed}/{len(FILES)} files. Memory freed.")

    # Post-Audit Analysis
    if global_infinity_counts:
        potential_issues.append("Infinity values detected in numeric columns.")

    if 'Label' in global_class_distribution:
        pass # Label is a class here if not cleaned properly

    # Check for Header Leaks specifically
    # We need to know if 'Label' class exists in distribution
    if 'Label' in global_class_distribution:
        leak_count = global_class_distribution['Label']
        potential_issues.append(f"Found {leak_count} header leak rows ('Label' as class).")
        # Remove from distribution for final report
        del global_class_distribution['Label']

    # Check CV Feasibility
    for cls, count in global_class_distribution.items():
        if count < CV_FOLDS:
            potential_issues.append(f"Class '{cls}' has {count} samples. Insufficient for {CV_FOLDS}-Fold CV.")

    report = {
        "files_processed": files_processed,
        "total_rows": total_rows,
        "estimated_memory_mb": total_memory_mb,
        "missing_values": global_missing_values,
        "infinity_counts": global_infinity_counts,
        "class_distribution": global_class_distribution,
        "potential_issues": potential_issues
    }

    return report

def validate_pipeline_sample(file_name: str) -> bool:
    """
    Validates pipeline on a single file sample to avoid OOM.
    """
    logger.info("Validating pipeline on sample data...")
    try:
        from src.data_loader import load_and_optimize_file
        df = load_and_optimize_file(file_name)
        if df is None:
            return False

        pipeline = get_cleaning_pipeline()
        # Sample 1000 rows
        sample = df.sample(n=min(1000, len(df)), random_state=42)
        cleaned_sample = pipeline.fit_transform(sample)

        logger.info(f"Pipeline validation successful. Output shape: {cleaned_sample.shape}")
        del df
        del sample
        del cleaned_sample
        gc.collect()
        return True
    except Exception as e:
        logger.error(f"Pipeline validation failed: {e}")
        return False

def run_stage1_audit():
    """
    Main execution function for Stage 1.
    """
    # 1. Run Iterative Audit
    audit_report = audit_iterative()

    # 2. Validate Pipeline on First Available File
    if FILES:
        pipeline_valid = validate_pipeline_sample(FILES[0])
        audit_report["pipeline_validation"] = "Passed" if pipeline_valid else "Failed"
    else:
        audit_report["pipeline_validation"] = "Skipped (No files)"

    # 3. Save Report
    report_path = REPORTS_DIR / "stage1_audit_report.json"
    with open(report_path, 'w') as f:
        json.dump(audit_report, f, indent=4, default=str)

    logger.info(f"Audit report saved to {report_path}")
    logger.info("=== Stage 1 Complete ===")

    return audit_report

if __name__ == "__main__":
    run_stage1_audit()
