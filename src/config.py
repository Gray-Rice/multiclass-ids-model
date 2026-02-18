import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "reports"
print(DATA_DIR)
# Ensure directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Files
FILES = [
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
]

# Modeling Constants
RANDOM_STATE = 42
CV_FOLDS = 5

# Known Non-Feature Columns (to be dropped)
DROP_COLUMNS = [
    "Flow ID", "Source IP", "Destination IP",
    "Source Port", "Destination Port", "Protocol"
]

# Logging Configuration
LOG_LEVEL = "INFO"
