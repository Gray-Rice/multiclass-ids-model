from src.explorer import run_stage1_audit

def main():
    # Stage 1: Audit
    report = run_stage1_audit()

    if report:
        print("\n--- Summary ---")
        print(f"Total Rows: {report['shape'][0]}")
        print(f"Potential Issues: {len(report['potential_issues'])}")
        for issue in report['potential_issues']:
            print(f" - {issue}")
        print(f"Pipeline Validation: {report['pipeline_validation']}")
        print("\nReview reports/stage1_audit_report.json for details.")

if __name__ == "__main__":
    main()
