# Intrusion Detection Model (CSE-CIC-IDS2018)

A practical ML pipeline for **multi-class network intrusion detection** using the CSE-CIC-IDS2018 dataset, with:
- preprocessing,
- tabular model training (LightGBM, RandomForest),
- LSTM baseline training,
- ablation analysis for leakage/shortcut checks.

## Features

- Unified preprocessing for train/val/test
- Multi-model training and metrics export
- LSTM sequence baseline
- Ablation runner for suspicious feature analysis
- JSON/CSV reports for easy comparison

---

## Project Structure

```text
src/
  config.py
  preprocess.py
  trainer_tabular.py
  trainer_lstm.py
  run_all.py
  ablation_runner.py
  rebuild_cleaned_files.py
data/
  raw/                # original IDS CSV files
  processed/          # cleaned_*.parquet, model artifacts
reports/              # metrics JSON/CSV outputs
```

---

## Setup

## 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
```

## 2) Install dependencies
```bash
pip install -U pip
pip install pandas numpy scikit-learn lightgbm pyarrow joblib tensorflow
```

> If TensorFlow GPU is not detected, training still works on CPU.

---

## Data Preparation

Place raw IDS CSV files in:
```text
data/raw/
```

Ensure `FILES` in `src/config.py` matches your dataset file names.

---

## Run Pipeline

## Rebuild cleaned per-file parquet files
```bash
python -m src.rebuild_cleaned_files
```

## Train tabular models
```bash
python -m src.trainer_tabular
```

## Train LSTM baseline
```bash
python -m src.trainer_lstm
```

## Run both + combined summary
```bash
python -m src.run_all
```

## Run ablation analysis
```bash
python -m src.ablation_runner
```

---

## Outputs

Generated artifacts are saved to:

- `data/processed/`
  - trained model files (`*.pkl`, `*.keras`)
  - encoders/scalers
  - selected feature lists
- `reports/`
  - `tabular_results.json`
  - `lstm_results.json`
  - `combined_results.json`
  - `ablation_results.json`
  - `ablation_results.csv`

---

## Notes on Evaluation

Current results can appear very high if split mode falls back to `row_order`.  
For reliable generalization claims, prefer **strict day-based hard splits** (`hard_split_runner.py`) with all day files available.

---

## Current Status

- Implementation complete through ablation stage
- Leakage-aware diagnostics integrated
- Hard split evaluation recommended as next step for final reporting

---

## Citation

If you use this repository in academic work, please cite:
- Sharafaldin et al., CSE-CIC-IDS2018
- LightGBM paper
- Random Forest paper
- LSTM original paper
