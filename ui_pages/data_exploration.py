from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def render_data_exploration(df: pd.DataFrame) -> None:
    st.header("Data Exploration")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Fraud Rate", f"{df['label_fraud'].mean():.2%}")
    c3.metric("Avg Amount", f"${df['amount'].mean():.2f}")

    fig1 = px.histogram(df, x="amount", nbins=60, color="label_fraud", barmode="overlay", title="Amount Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fraud_by_type = df.groupby("transaction_type", as_index=False)["label_fraud"].mean()
    fig2 = px.bar(fraud_by_type, x="transaction_type", y="label_fraud", title="Fraud Rate by Transaction Type")
    st.plotly_chart(fig2, use_container_width=True)

    df_time = df.copy()
    df_time["timestamp"] = pd.to_datetime(df_time["timestamp"])
    df_time["hour"] = df_time["timestamp"].dt.hour
    by_hour = df_time.groupby("hour", as_index=False)["label_fraud"].mean()
    fig3 = px.line(by_hour, x="hour", y="label_fraud", title="Fraud Rate by Hour")
    st.plotly_chart(fig3, use_container_width=True)
