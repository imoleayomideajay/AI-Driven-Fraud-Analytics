from __future__ import annotations

import streamlit as st


def render_home() -> None:
    st.title("🏦 AI-Driven Fraud Analytics Platform")
    st.markdown(
        """
        A portfolio-grade fraud monitoring prototype for digital banking transactions.

        **Capabilities**
        - Synthetic transaction simulation with realistic fraud patterns
        - ML training and champion model comparison
        - Live scoring + batch scoring workflow
        - Explainability for fraud analysts
        - Monitoring and drift indicators for production readiness
        """
    )

    st.info(
        "Use the sidebar to navigate: Data Exploration → Model Performance → Live Scoring → Batch Scoring → Explainability → Monitoring."
    )
