"""Streamlit app entrypoint for AI-driven fraud analytics."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from pages.batch_scoring import render_batch_scoring
from pages.data_exploration import render_data_exploration
from pages.explainability import render_explainability
from pages.home import render_home
from pages.live_scoring import render_live_scoring
from pages.model_performance import render_model_performance
from pages.monitoring import render_monitoring
from src.data_simulation import SimulationConfig, generate_synthetic_transactions, save_dataset
from src.predict import load_model_artifacts
from src.train import run_training
from src.utils import DATA_DIR, ensure_directories

logging.basicConfig(level=logging.INFO)


def _load_or_generate_data() -> pd.DataFrame:
    data_path = DATA_DIR / "synthetic_transactions.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    df = generate_synthetic_transactions(SimulationConfig())
    save_dataset(df, data_path)
    return df


@st.cache_resource(show_spinner=False)
def _ensure_model_ready() -> tuple[object | None, dict]:
    model_path = Path("models/champion_model.joblib")
    metadata_path = Path("models/training_metadata.json")

    if model_path.exists() and metadata_path.exists():
        return load_model_artifacts()

    try:
        from src.train import run_training

        run_training(n_rows=25000)
        return load_model_artifacts()
    except Exception as exc:
        logger.exception("Model training/loading failed: %s", exc)
        return None, {
            "champion_model": "rules_fallback",
            "champion_threshold": 0.5,
            "metrics": {},
            "error": str(exc),
        }


def main() -> None:
    st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="🏦")
    ensure_directories()

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

    data = _load_or_generate_data()
    model, metadata = _ensure_model_ready()
    threshold = float(metadata.get("champion_threshold", 0.5))

    st.sidebar.markdown("---")
    st.sidebar.metric("Train Fraud Rate", f"{data['label_fraud'].mean():.2%}")
    st.sidebar.metric("Champion", metadata.get("champion_model", "N/A"))

    if page == "Home":
        render_home()
    elif page == "Data Exploration":
        render_data_exploration(data)
    elif page == "Model Performance":
        render_model_performance(metadata)
    elif page == "Live Scoring":
        render_live_scoring(model, threshold)
    elif page == "Batch Scoring":
        render_batch_scoring(model, threshold)
    elif page == "Explainability":
        render_explainability(model, data)
    elif page == "Monitoring":
        render_monitoring(data, model, threshold)


if __name__ == "__main__":
    main()
