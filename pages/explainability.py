from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.explain import global_feature_importance



def render(df: pd.DataFrame) -> None:
    st.header("Explainability")
    imp = global_feature_importance(df)
    top_imp = imp.head(20)
    fig = px.bar(top_imp.sort_values("importance"), x="importance", y="feature", orientation="h", title="Global Fraud Drivers")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top_imp, use_container_width=True)
