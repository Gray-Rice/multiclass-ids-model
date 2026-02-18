import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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

RANDOM_STATE = 42
CV_FOLDS = 3

DROP_COLUMNS = [
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port"
]

LABEL_MAPPING = {
    "Infilteration": "Infiltration",
    "DDOS attack-HOIC": "DDoS attack-HOIC",
    "DDOS attack-LOIC-UDP": "DDoS attack-LOIC-UDP",
    "DDoS attacks-LOIC-HTTP": "DDoS attack-LOIC-HTTP",
}

MAX_ROWS = 2000000
N_JOBS = 2

LOG_LEVEL = "INFO"
