from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import score_transactions, simple_drift_check


def render_monitoring(train_df: pd.DataFrame, model, threshold: float) -> None:
    st.header("Monitoring")
    uploaded = st.file_uploader("Upload new transaction batch for monitoring", type=["csv"], key="monitor_upload")

    if uploaded is None:
        st.info("Upload a new batch to see fraud rate trends, score distributions, and drift checks.")
        return

    new_df = pd.read_csv(uploaded)
    scored = score_transactions(new_df, model, threshold)

    c1, c2, c3 = st.columns(3)
    c1.metric("New Batch Fraud Rate", f"{scored['fraud_prediction'].mean():.2%}")
    c2.metric("Avg Fraud Score", f"{scored['fraud_probability'].mean():.2%}")
    c3.metric("High Risk Alerts", int((scored["risk_label"] == "High").sum()))

    fig = px.histogram(scored, x="fraud_probability", nbins=50, title="Fraud Score Distribution")
    st.plotly_chart(fig, use_container_width=True)

    drift = simple_drift_check(train_df, new_df)
    st.subheader("Simple Drift Check (Train vs New Batch)")
    st.dataframe(drift, use_container_width=True)

    st.subheader("Model Monitoring Summary")
    significant = drift[drift["delta_pct"].abs() > 0.25]
    if significant.empty:
        st.success("No major drift indicator found above 25% mean shift threshold.")
    else:
        st.warning(f"Potential drift detected in {len(significant)} features. Review before model promotion.")
        st.dataframe(significant, use_container_width=True)
