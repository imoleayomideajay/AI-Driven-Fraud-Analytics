from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.explain import global_feature_importance, shap_values_if_available


def render_explainability(model, sample_df) -> None:
    st.header("Explainability")
    if sample_df.empty:
        st.warning("No sample data available for explainability.")
        return

    try:
        importance = global_feature_importance(model, sample_df.head(2000))
    except Exception as exc:
        st.error(f"Explainability failed: {exc}")
        st.info("Tip: ensure the uploaded/scored data schema includes the core transaction fields.")
        return

    top = importance.head(20)
    fig = px.bar(top.iloc[::-1], x="importance", y="feature", orientation="h", title="Top Global Fraud Drivers")
    st.plotly_chart(fig, use_container_width=True)

    shap_out = shap_values_if_available(model, sample_df.head(250))
    if shap_out.get("available"):
        st.success("SHAP backend is available. Integrate SHAP plots for local-level explanations.")
    else:
        st.caption("SHAP unavailable in runtime; showing robust model-native feature importance fallback.")
