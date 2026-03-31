from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st



def render(df: pd.DataFrame) -> None:
    st.header("Data Exploration")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Fraud Rate", f"{df['label_fraud'].mean():.2%}")
    c3.metric("Avg Amount", f"${df['amount'].mean():.2f}")

    fig1 = px.histogram(df, x="amount", color="label_fraud", nbins=60, title="Transaction Amount Distribution")
    fig2 = px.box(df.sample(min(12000, len(df)), random_state=42), x="transaction_type", y="velocity_score", color="label_fraud")
    fig3 = px.histogram(df, x=df["timestamp"].dt.hour, color="label_fraud", nbins=24, title="Fraud by Hour")

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
