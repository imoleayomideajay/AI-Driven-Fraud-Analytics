# AI-Driven Fraud Analytics (Bank-Grade Prototype)

## Project Overview
This project is an end-to-end fraud detection system for a digital bank / fintech context. It includes:
- realistic synthetic transaction simulation (50k+ rows),
- feature engineering for fraud signals,
- model training and comparison for imbalanced classification,
- explainability outputs,
- production-style scoring utilities,
- a multi-page Streamlit dashboard for fraud operations and monitoring.

## Project Architecture
```text
AI-Driven-Fraud-Analytics/
├── app.py
├── ui_pages/
│   ├── home.py
│   ├── data_exploration.py
│   ├── model_performance.py
│   ├── live_scoring.py
│   ├── batch_scoring.py
│   ├── explainability.py
│   └── monitoring.py
├── src/
│   ├── data_simulation.py
│   ├── features.py
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── data/
├── models/
├── outputs/
├── requirements.txt
└── README.md
```

## Setup Instructions (Local)
1. Clone repo and enter directory.
2. Create a virtual env and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Train models and produce artifacts:
   ```bash
   python -m src.train
   ```
4. Run Streamlit app:
   ```bash
   streamlit run app.py
   ```

## How to Train
`python -m src.train` will:
- simulate transactions with embedded fraud behavior,
- train Logistic Regression, Random Forest, Gradient Boosting (and XGBoost only if available), and Isolation Forest benchmark,
- tune threshold for F1,
- evaluate fraud metrics (precision/recall/F1/ROC-AUC/PR-AUC/FPR/capture@top5%),
- persist champion model and metadata in `models/`.

## Streamlit Sections
Single top navigation bar controls all sections.

1. **Home / Overview**
2. **Data Exploration**
3. **Model Performance**
4. **Live Scoring** (single transaction)
5. **Batch Scoring** (CSV upload + download)
6. **Explainability** (global feature importance)
7. **Monitoring** (fraud rate, score distribution, mean-shift drift, PSI stability index, alerts)

## Fraud Scenarios the App Detects
- High-velocity bursts (many transactions in 24h + high velocity score).
- Anomalous geo behavior (very large distance from home).
- High-risk web/mobile + unknown-device combinations.
- New beneficiary fraud spikes.
- Amount anomalies relative to 7-day customer baseline.
- Night-time concentration and higher-risk transaction type patterns.

## Model Monitoring Summary
The monitoring page includes:
- current batch fraud rate and high-risk alert count,
- fraud score distribution,
- simple drift check comparing training vs uploaded batch means,
- warning when mean feature shift exceeds a 25% threshold.
- PSI-based stability classification (`stable`, `moderate`, `high`) for retraining governance.

## Deployment on Streamlit Community Cloud
1. Push repository to GitHub.
2. On Streamlit Community Cloud, create a new app from the repo.
3. Set main file to `app.py`.
4. Ensure `requirements.txt` is in root.
5. Deploy.

## Deployment Notes
- Paths are relative and deployment-friendly.
- If model artifacts are missing, `app.py` attempts training and gracefully falls back to deterministic rules-based scoring if runtime dependencies are constrained.
- XGBoost and SHAP are optional enhancements; the app runs without them.

## Sample Screenshot Placeholders (to capture after running)
- `docs/screenshots/home.png`: Home overview and KPI sidebar.
- `docs/screenshots/model_performance.png`: model comparison charts.
- `docs/screenshots/monitoring.png`: drift table and alert summary.

> Note: Screenshots are described here for reproducibility and should be captured in your local runtime or CI visual artifact workflow.
