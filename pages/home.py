"""Home page for fraud analytics dashboard."""

from __future__ import annotations

import streamlit as st


def render_home() -> None:
    st.title("🏦 AI-Driven Fraud Detection Command Center")
    st.markdown(
        """
        This prototype simulates retail banking transactions and detects suspicious activity using supervised and anomaly models.

        **Capabilities**
        - Synthetic transaction generation with realistic fraud signatures.
        - Model training and evaluation for imbalanced fraud data.
        - Live and batch scoring with rules + ML risk labels.
        - Explainability and monitoring with drift-style indicators.
        """
    )
