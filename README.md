# AI-Driven Fraud Analytics (Bank-Grade Prototype)

A complete end-to-end fraud detection system for digital banking/card/transfer monitoring using synthetic data and a production-style Python architecture.

## 1) Project architecture

```text
AI-Driven-Fraud-Analytics/
├── app.py
├── data/
├── models/
├── outputs/
├── pages/
│   ├── __init__.py
│   ├── home.py
│   ├── data_exploration.py
│   ├── model_performance.py
│   ├── live_scoring.py
│   ├── batch_scoring.py
│   ├── explainability.py
│   └── monitoring.py
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── data_simulation.py
│   ├── features.py
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── train.py
│   ├── explain.py
│   ├── predict.py
│   └── monitoring.py
├── requirements.txt
└── README.md
```

## 2) Key capabilities

- Synthetic transaction generation (`>=50,000 rows`) with realistic fraud patterns:
  - high velocity bursts
  - unusual geolocation behavior
  - risky device/channel combinations
  - rapid beneficiary creation
  - anomalous amount vs personal baseline
  - fraud concentration by transaction type and time-of-day
- Feature engineering and leakage-safe train/test split.
- Model benchmark:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting (XGBoost-equivalent fallback)
  - Isolation Forest anomaly benchmark
- Imbalanced metrics:
  - precision, recall, F1, ROC-AUC, PR-AUC
  - confusion-derived false positive rate
  - fraud capture rate (recall@top-5% risk)
- Threshold tuning (max F1) and model champion selection.
- Explainability:
  - model-based feature importance
  - SHAP local explanations when feasible
  - robust fallback if SHAP path fails
- Streamlit dashboard pages:
  1. Home/Overview
  2. Data Exploration
  3. Model Performance
  4. Live Scoring
  5. Batch Scoring + CSV export
  6. Explainability
  7. Monitoring + drift check (PSI)
- Fraud alert rule panel shown together with ML output.

## 3) Setup instructions

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## 4) Train pipeline

```bash
python -m src.train
```

This will:
- generate synthetic dataset (`data/synthetic_transactions.csv`)
- train all models and evaluate
- save metrics to `outputs/model_metrics.csv`
- save model artifacts to `models/`

## 5) Run Streamlit app

```bash
streamlit run app.py
```

The app auto-generates dataset and artifacts if missing.

## 6) Deployment (GitHub + Streamlit Community Cloud)

1. Push repo to GitHub.
2. In Streamlit Community Cloud, create new app from repo.
3. Set entry file to `app.py`.
4. Ensure `requirements.txt` is in root.
5. Deploy.

No absolute paths are used; deployment is relative-path friendly.

## 7) Sample fraud scenarios detected

1. **Velocity attack**: 12 transactions in 24h + failed logins + new beneficiary.
2. **Account takeover signal**: web channel + emulator + high IP risk + night-time transfer.
3. **Mule payout behavior**: bank transfer amount 5x historical average with 400km geo deviation.
4. **Burst cash-out**: cash withdrawals after repeated login failures from risky device.

## 8) Monitoring summary + drift checks

Monitoring page provides:
- predicted fraud rate and high-risk rate
- alert density (rule alerts per transaction)
- fraud score distribution
- PSI-based drift check (training vs uploaded batch)

PSI guidance:
- `<0.10` Stable
- `0.10–0.25` Moderate drift
- `>0.25` Significant drift

## 9) Sample screenshot descriptions (for stakeholder demo)

- **Home page**: executive summary with navigation and KPI sidebar.
- **Model performance page**: benchmark table and metric comparison bars.
- **Live scoring page**: transaction input form, fraud score, risk label, and rules panel.
- **Monitoring page**: score histogram, rule alert histogram, and PSI drift table.

(If you run the app locally, capture screenshots and place them under `outputs/screenshots/`.)
