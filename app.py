"""Streamlit app entrypoint for AI-driven fraud analytics."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_simulation import SimulationConfig, generate_synthetic_transactions, save_dataset
from src.predict import load_model_artifacts
from src.utils import DATA_DIR, ensure_directories
from ui_pages.batch_scoring import render_batch_scoring
from ui_pages.data_exploration import render_data_exploration
from ui_pages.explainability import render_explainability
from ui_pages.home import render_home
from ui_pages.live_scoring import render_live_scoring
from ui_pages.model_performance import render_model_performance
from ui_pages.monitoring import render_monitoring

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@st.cache_data(show_spinner=False)
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


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {background: linear-gradient(180deg,#f6f9ff 0%,#edf3ff 100%);} 
        .block-container {padding-top: 1.2rem; padding-bottom: 1.5rem; max-width: 1400px;}
        .hero {background: linear-gradient(90deg,#0b1f4d,#133b8a); padding: 18px 24px; border-radius: 14px; color: #fff; margin-bottom: 12px;}
        .hero h1 {font-size: 1.6rem; margin: 0;}
        .hero p {margin: 6px 0 0 0; opacity: 0.92;}
        .stTabs [data-baseweb="tab-list"] {gap: 10px;}
        .stTabs [data-baseweb="tab"] {height: 44px; border-radius: 10px; background:#e9efff;}
        [data-testid="stMetric"] {background: #ffffff; border: 1px solid #e7ebf3; padding: 10px 12px; border-radius: 10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="🏦")
    ensure_directories()
    _inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>🏦 Fraud Risk Command Center</h1>
            <p>Industry-grade transaction monitoring with ML + deterministic AML rule intelligence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data = _load_or_generate_data()
    model, metadata = _ensure_model_ready()
    threshold = float(metadata.get("champion_threshold", 0.5))

    if metadata.get("champion_model") == "rules_fallback":
        st.warning("ML artifacts unavailable. Running robust rules-based fallback scoring.")

    # Single navigation bar (top horizontal)
    page = st.radio(
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
        horizontal=True,
        label_visibility="collapsed",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(data):,}")
    c2.metric("Fraud Rate", f"{data['label_fraud'].mean():.2%}" if "label_fraud" in data.columns else "N/A")
    c3.metric("Champion Model", metadata.get("champion_model", "N/A"))
    c4.metric("Decision Threshold", f"{threshold:.2f}")

    st.markdown("---")

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
        if model is None:
            st.info("Explainability is unavailable in rules fallback mode. Train/load ML artifacts to enable this page.")
        else:
            render_explainability(model, data)
    elif page == "Monitoring":
        render_monitoring(data, model, threshold)


if __name__ == "__main__":
    main()
