from __future__ import annotations

import streamlit as st



def render() -> None:
    st.title("🏦 AI-Driven Fraud Detection Control Center")
    st.markdown(
        """
        This portfolio-grade prototype simulates a digital bank fraud stack:
        - Synthetic transaction generation with realistic fraud behavior.
        - Feature engineering for risk signals.
        - Multi-model benchmarking and threshold tuning.
        - Live + batch scoring with explainability and monitoring.
        """
    )
    st.info("Use the left sidebar to navigate pages.")
