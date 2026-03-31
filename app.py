"""Streamlit app entrypoint for fraud analytics prototype."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pages import batch_scoring, data_exploration, explainability, home, live_scoring, model_performance, monitoring
from src.data_simulation import create_and_save_dataset
from src.train import train_and_evaluate
from src.utils import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, ensure_directories

st.set_page_config(page_title="Fraud Detection Control Center", page_icon="🏦", layout="wide")


@st.cache_data(show_spinner=False)
def load_or_create_data() -> pd.DataFrame:
    ensure_directories()
    path = DATA_DIR / "synthetic_transactions.csv"
    if not path.exists():
        df, _ = create_and_save_dataset(n_rows=60_000)
        return df
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=True)
def ensure_trained_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    metrics_path = OUTPUTS_DIR / "model_metrics.csv"
    if not (MODELS_DIR / "champion_pipeline.joblib").exists() or not metrics_path.exists():
        train_and_evaluate(df)
    return pd.read_csv(metrics_path, index_col=0)



def main() -> None:
    df = load_or_create_data()
    metrics_df = ensure_trained_artifacts(df)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Data Exploration",
            "Model Performance",
            "Live Scoring",
            "Batch Scoring",
            "Explainability",
            "Monitoring",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.metric("Transactions", f"{len(df):,}")
    st.sidebar.metric("Observed Fraud Rate", f"{df['label_fraud'].mean():.2%}")

    if page == "Home":
        home.render()
    elif page == "Data Exploration":
        data_exploration.render(df)
    elif page == "Model Performance":
        model_performance.render(metrics_df)
    elif page == "Live Scoring":
        live_scoring.render()
    elif page == "Batch Scoring":
        batch_scoring.render()
    elif page == "Explainability":
        explainability.render(df)
    else:
        monitoring.render(df)


if __name__ == "__main__":
    main()
