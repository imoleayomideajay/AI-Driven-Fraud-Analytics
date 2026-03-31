from __future__ import annotations

import pandas as pd
import streamlit as st

from src.predict import generate_rule_alerts, score_transactions



def render() -> None:
    st.header("Batch Scoring")
    uploaded = st.file_uploader("Upload CSV transactions", type=["csv"])
    if uploaded is None:
        st.caption("Upload a file with the same schema as synthetic data (minus label_fraud is fine).")
        return

    df = pd.read_csv(uploaded)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    scored = generate_rule_alerts(score_transactions(df))
    st.dataframe(scored.head(200), use_container_width=True)
    st.download_button(
        "Download scored CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="scored_transactions.csv",
        mime="text/csv",
    )
