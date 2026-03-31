from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from src.predict import score_dataframe



def render_batch_scoring(pipeline, threshold, feature_columns) -> None:
    st.header("Batch Scoring")
    file = st.file_uploader("Upload CSV for batch scoring", type=["csv"])

    if file is None:
        st.info("Upload a CSV with transaction schema to score records.")
        return

    batch_df = pd.read_csv(file)
    if "label_fraud" not in batch_df.columns:
        batch_df["label_fraud"] = 0

    scored = score_dataframe(batch_df, pipeline, threshold, feature_columns)
    st.dataframe(scored.head(100))

    c1, c2, c3 = st.columns(3)
    c1.metric("Batch Rows", len(scored))
    c2.metric("Flagged", int(scored["predicted_fraud"].sum()))
    c3.metric("High Risk", int((scored["risk_band"] == "High").sum()))

    csv_bytes = scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Scored Output",
        data=BytesIO(csv_bytes),
        file_name="scored_transactions.csv",
        mime="text/csv",
    )
