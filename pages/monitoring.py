from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.monitoring import drift_summary, monitoring_kpis
from src.predict import generate_rule_alerts, score_transactions



def render(train_df: pd.DataFrame) -> None:
    st.header("Monitoring")
    uploaded = st.file_uploader("Upload recent production-like transactions", type=["csv"], key="monitor_upload")
    if uploaded is None:
        st.caption("Upload a recent batch to compute fraud KPIs, score distributions, and drift signals.")
        return

    new_df = pd.read_csv(uploaded)
    if "timestamp" in new_df.columns:
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])

    scored = generate_rule_alerts(score_transactions(new_df))
    kpis = monitoring_kpis(scored)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Fraud Rate", f"{kpis['fraud_rate']:.2%}")
    c2.metric("High Risk Rate", f"{kpis['high_risk_rate']:.2%}")
    c3.metric("Average Score", f"{kpis['avg_score']:.3f}")
    c4.metric("Avg Rule Alerts/Txn", f"{kpis['alerts_per_tx']:.2f}")

    st.plotly_chart(px.histogram(scored, x="fraud_score", nbins=40, title="Fraud Score Distribution"), use_container_width=True)
    st.plotly_chart(px.histogram(scored, x="rule_alert_count", nbins=10, title="Rule Alert Count Distribution"), use_container_width=True)

    numeric_cols = [
        "amount",
        "ip_risk_score",
        "geo_distance_from_home",
        "transaction_count_24h",
        "failed_login_count_24h",
        "velocity_score",
    ]
    drift_df = drift_summary(train_df, new_df, numeric_cols)
    st.subheader("Simple Drift Check (PSI)")
    st.dataframe(drift_df, use_container_width=True)
