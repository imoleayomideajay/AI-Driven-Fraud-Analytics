from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.monitoring import psi_summary
from src.predict import score_transactions, simple_drift_check


MONITOR_FEATURES = [
    "amount",
    "ip_risk_score",
    "geo_distance_from_home",
    "transaction_count_24h",
    "velocity_score",
]


def render_monitoring(train_df: pd.DataFrame, model, threshold: float) -> None:
    st.header("Monitoring")
    uploaded = st.file_uploader("Upload new transaction batch for monitoring", type=["csv"], key="monitor_upload")

    if uploaded is None:
        st.info("Upload a new batch to see fraud rate trends, score distributions, and drift checks.")
        return

    new_df = pd.read_csv(uploaded)
    scored = score_transactions(new_df, model, threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("New Batch Fraud Rate", f"{scored['fraud_prediction'].mean():.2%}")
    c2.metric("Avg Fraud Score", f"{scored['fraud_probability'].mean():.2%}")
    c3.metric("High Risk Alerts", int((scored["risk_label"] == "High").sum()))
    c4.metric("Rules Avg Triggered", f"{scored['rules_triggered'].mean():.2f}")

    fig = px.histogram(scored, x="fraud_probability", nbins=50, title="Fraud Score Distribution")
    st.plotly_chart(fig, use_container_width=True)

    drift = simple_drift_check(train_df, new_df)
    st.subheader("Mean Shift Drift Check (Train vs New Batch)")
    st.dataframe(drift, use_container_width=True)

    st.subheader("Population Stability Index (PSI)")
    psi_df = psi_summary(train_df, new_df, MONITOR_FEATURES)
    st.dataframe(psi_df, use_container_width=True)

    st.subheader("Model Monitoring Summary")
    significant = drift[drift["delta_pct"].abs() > 0.25]
    high_psi = psi_df[psi_df["severity"] == "high"] if not psi_df.empty else pd.DataFrame()

    if significant.empty and high_psi.empty:
        st.success("No major drift indicators detected.")
    else:
        if not significant.empty:
            st.warning(f"Mean-shift drift detected in {len(significant)} features.")
            st.dataframe(significant, use_container_width=True)
        if not high_psi.empty:
            st.error(f"High PSI detected in {len(high_psi)} features. Consider retraining.")
            st.dataframe(high_psi, use_container_width=True)
