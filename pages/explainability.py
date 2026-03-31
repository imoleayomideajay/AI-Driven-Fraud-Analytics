from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.explain import global_feature_importance


def render_explainability(model, sample_df) -> None:
    st.header("Explainability")
    if sample_df.empty:
        st.warning("No sample data available for explainability.")
        return

    importance = global_feature_importance(model, sample_df.head(2000))
    top = importance.head(20)
    fig = px.bar(top.iloc[::-1], x="importance", y="feature", orientation="h", title="Top Global Fraud Drivers")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("If SHAP is installed, extend this page with local SHAP force/waterfall plots.")
