from __future__ import annotations

import json
from pathlib import Path

import plotly.express as px
import streamlit as st


def render_model_performance(outputs_dir: Path, models_dir: Path) -> None:
    st.header("Model Performance")

    comparison_path = outputs_dir / "model_comparison.csv"
    summary_path = models_dir / "train_summary.json"

    if not comparison_path.exists() or not summary_path.exists():
        st.warning("Training artifacts not found. Please run `python -m src.train` first.")
        return

    df = __import__("pandas").read_csv(comparison_path)
    with open(summary_path, "r", encoding="utf-8") as fp:
        summary = json.load(fp)

    st.success(f"Champion model: **{summary['champion_model']}**")
    st.dataframe(df)

    metric_choice = st.selectbox("Compare metric", ["pr_auc", "roc_auc", "f1", "recall", "precision"])
    st.plotly_chart(px.bar(df, x="model_name", y=metric_choice, title=f"Model comparison by {metric_choice}"), use_container_width=True)

    st.caption(f"Selected threshold: {summary['champion_metrics']['threshold']:.3f}")
