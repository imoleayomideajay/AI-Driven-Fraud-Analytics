from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.monitoring import compute_monitoring_summary, numeric_drift_report
from src.predict import score_dataframe



def render_monitoring(train_df, pipeline, threshold, feature_columns) -> None:
    st.header("Monitoring & Drift")
    upload = st.file_uploader("Upload new transaction batch for monitoring", type=["csv"], key="monitor_upload")

    if upload is None:
        st.info("Upload a recent production-like batch to view monitoring KPIs and drift checks.")
        return

    new_df = __import__("pandas").read_csv(upload)
    if "label_fraud" not in new_df.columns:
        new_df["label_fraud"] = 0

    scored = score_dataframe(new_df, pipeline, threshold, feature_columns)
    summary = compute_monitoring_summary(scored)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", summary["transactions"])
    c2.metric("Flagged", summary["flagged"])
    c3.metric("Flag Rate", f"{summary['flag_rate']:.2%}")
    c4.metric("High Risk", summary["high_risk"])

    st.subheader("Score Distribution")
    st.plotly_chart(px.histogram(scored, x="fraud_probability", nbins=40, color="risk_band"), use_container_width=True)

    st.subheader("Simple Drift Check (Training vs Uploaded Batch)")
    drift_cols = [
        "amount",
        "ip_risk_score",
        "geo_distance_from_home",
        "transaction_count_24h",
        "failed_login_count_24h",
        "velocity_score",
    ]
    drift_df = numeric_drift_report(train_df, new_df, columns=drift_cols)
    st.dataframe(drift_df)

    if len(drift_df) > 0:
        st.plotly_chart(px.bar(drift_df, x="feature", y="psi", color="drift_flag", title="PSI Drift by Feature"), use_container_width=True)
