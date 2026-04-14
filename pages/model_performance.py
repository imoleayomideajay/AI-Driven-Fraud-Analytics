from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def render_model_performance(metadata: dict) -> None:
    st.header("Model Performance")
    st.subheader(f"Champion Model: {metadata.get('champion_model', 'N/A')}")
    st.write(f"Decision threshold: **{metadata.get('champion_threshold', 0.5):.2f}**")
    st.caption("Threshold policy: maximize recall under a minimum precision floor for operationally efficient alerting.")

    metrics_rows = []
    for model_name, values in metadata.get("metrics", {}).items():
        row = {"model": model_name}
        row.update(values)
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    st.dataframe(metrics_df, use_container_width=True)

    if not metrics_df.empty:
        fig = px.bar(
            metrics_df,
            x="model",
            y=["precision", "recall", "f1", "roc_auc", "pr_auc"],
            barmode="group",
            title="Model Metric Comparison",
        )
        st.plotly_chart(fig, use_container_width=True)

        heat_df = metrics_df.set_index("model")[["precision", "recall", "f1", "roc_auc", "pr_auc"]]
        heat = px.imshow(heat_df, color_continuous_scale="Blues", aspect="auto", title="Metric Heatmap")
        st.plotly_chart(heat, use_container_width=True)
