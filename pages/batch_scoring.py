from __future__ import annotations

import pandas as pd
import streamlit as st

from src.predict import score_transactions


def render_batch_scoring(model, threshold: float) -> None:
    st.header("Batch Scoring")
    uploaded = st.file_uploader("Upload CSV for scoring", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with transaction fields to generate fraud risk scores.")
        return

    df = pd.read_csv(uploaded)
    scored = score_transactions(df, model, threshold)

    st.dataframe(scored.head(100), use_container_width=True)
    st.download_button(
        "Download Scored CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="scored_transactions.csv",
        mime="text/csv",
    )
