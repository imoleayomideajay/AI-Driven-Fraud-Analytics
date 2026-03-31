from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st



def render(metrics_df: pd.DataFrame) -> None:
    st.header("Model Performance")
    st.dataframe(metrics_df, use_container_width=True)

    plot_df = metrics_df.reset_index().rename(columns={"index": "model"})
    fig = px.bar(plot_df, x="model", y=["precision", "recall", "f1", "pr_auc", "roc_auc"], barmode="group")
    st.plotly_chart(fig, use_container_width=True)
