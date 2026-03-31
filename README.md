# AI-Driven Fraud Analytics (Digital Banking Prototype)

A production-style, portfolio-quality Python project that simulates digital banking transactions and builds an end-to-end fraud detection system with explainability and monitoring in a deployable Streamlit application.

## Project Architecture

```text
AI-Driven-Fraud-Analytics/
├── app.py
├── data/
│   └── synthetic_transactions.csv                # generated
├── models/
│   ├── champion_pipeline.joblib                  # generated
│   ├── champion_threshold.joblib                 # generated
│   ├── feature_columns.joblib                    # generated
│   └── train_summary.json                        # generated
├── outputs/
│   └── model_comparison.csv                      # generated
├── pages/
│   ├── home.py
│   ├── data_exploration.py
│   ├── model_performance.py
│   ├── live_scoring.py
│   ├── batch_scoring.py
│   ├── explainability.py
│   └── monitoring.py
├── src/
│   ├── data_simulation.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── predict.py
│   ├── monitoring.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## What the System Does

1. **Generates synthetic transaction data** with realistic fraud behaviors.
2. **Performs feature engineering** for fraud-relevant patterns.
3. **Trains and compares models**:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting (XGBoost-equivalent fallback)
   - Isolation Forest benchmark
4. **Handles class imbalance** using class weighting and threshold tuning.
5. **Evaluates with fraud-focused metrics** (precision, recall, F1, ROC-AUC, PR-AUC, FPR, top-5% capture).
6. **Supports explainability** via feature importance and SHAP fallback reason codes.
7. **Deploys as a Streamlit dashboard** with live + batch scoring, monitoring, and drift checks.

## Setup Instructions (Local)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Generate Data

```bash
python -m src.data_simulation
```

- Produces at least **60,000 transactions** in `data/synthetic_transactions.csv`.

## Train Models

```bash
python -m src.train
```

Outputs:
- `outputs/model_comparison.csv`
- `models/champion_pipeline.joblib`
- `models/champion_threshold.joblib`
- `models/feature_columns.joblib`
- `models/train_summary.json`

## Run Streamlit App

```bash
streamlit run app.py
```

## Streamlit App Sections

1. **Home / Overview**
2. **Data Exploration** (fraud patterns and distribution views)
3. **Model Performance** (model comparison + champion)
4. **Live Scoring** (single transaction scoring + alert rules panel)
5. **Batch Scoring** (CSV upload + downloadable scored output)
6. **Explainability** (global drivers + SHAP/fallback local reasons)
7. **Monitoring** (fraud KPIs + score distribution + PSI drift checks)

## Fraud Scenarios This App Detects

- High-velocity transaction bursts in 24h.
- Unusual geolocation far from customer home behavior.
- Risky device/channel combinations (e.g., emulator + API/web).
- Rapid beneficiary setup followed by transfer attempts.
- Transaction amounts anomalously high vs 7-day customer baseline.
- Elevated fraud concentration in riskier transaction types and night hours.

## Deployment on GitHub + Streamlit Community Cloud

### GitHub
1. Push this project to a GitHub repository.
2. Ensure generated artifacts exist or provide scripts to generate them.
3. Commit `requirements.txt` and all source files.

### Streamlit Cloud
1. Go to [https://share.streamlit.io](https://share.streamlit.io).
2. Connect your GitHub repo.
3. Set main file path to `app.py`.
4. Deploy.

> Recommended: pre-generate model artifacts (`models/`) and data file (`data/`) before deployment for a smoother first run.

## Monitoring Notes

- Monitoring page computes a **lightweight drift signal** using Population Stability Index (PSI) between training and uploaded batch numeric features.
- Drift bands:
  - Low: PSI < 0.10
  - Moderate: 0.10 ≤ PSI < 0.25
  - High: PSI ≥ 0.25

## Sample Screenshot Descriptions

- **Home Dashboard**: executive overview with navigation and capability summary.
- **Model Performance**: bar chart comparing PR-AUC, ROC-AUC, recall, precision across models.
- **Live Scoring**: single transaction score with ML probability, risk band, and triggered fraud alert rules.
- **Monitoring**: uploaded batch KPIs, score distribution histogram, and drift table with PSI.

(If running locally, capture these pages and add actual images under a `docs/images/` folder for portfolio presentation.)
