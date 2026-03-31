"""Streamlit app entry point."""

from __future__ import annotations

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
from src.predict import load_artifacts

st.set_page_config(page_title="Fraud Analytics", layout="wide", page_icon="🏦")

DATA_PATH = Path("data/synthetic_transactions.csv")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")


@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Dataset missing. Run `python -m src.data_simulation`.")
    return pd.read_csv(path, parse_dates=["timestamp"])


@st.cache_resource(show_spinner=False)
def load_model_artifacts(models_dir: Path):
    return load_artifacts(models_dir)



def main() -> None:
    st.sidebar.title("Fraud Analytics")
    page = st.sidebar.radio(
        "Navigation",
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

    try:
        df = load_dataset(DATA_PATH)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if page in {"Live Scoring", "Batch Scoring", "Explainability", "Monitoring"}:
        try:
            pipeline, threshold, feature_columns = load_model_artifacts(MODELS_DIR)
        except FileNotFoundError:
            st.error("Model artifacts missing. Run `python -m src.train` first.")
            st.stop()

    if page == "Home":
        render_home()
    elif page == "Data Exploration":
        render_data_exploration(df)
    elif page == "Model Performance":
        render_model_performance(OUTPUTS_DIR, MODELS_DIR)
    elif page == "Live Scoring":
        render_live_scoring(pipeline, threshold, feature_columns)
    elif page == "Batch Scoring":
        render_batch_scoring(pipeline, threshold, feature_columns)
    elif page == "Explainability":
        render_explainability(df, pipeline, feature_columns)
    elif page == "Monitoring":
        render_monitoring(df, pipeline, threshold, feature_columns)


if __name__ == "__main__":
    main()
