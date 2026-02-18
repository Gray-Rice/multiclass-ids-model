from src.explorer import run_stage1_audit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        report = run_stage1_audit()

        if report:
            print("\n--- Audit Summary ---")
            print(f"Total Rows: {report['total_rows']}")
            print(f"Files Processed: {report['files_processed']}")
            print(f"Potential Issues: {len(report['potential_issues'])}")
            for issue in report['potential_issues']:
                print(f" - {issue}")
            print(f"Pipeline Validation: {report['pipeline_validation']}")
            print("\nReview reports/stage1_audit_report.json for details.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
