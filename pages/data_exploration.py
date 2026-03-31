from __future__ import annotations

import plotly.express as px
import streamlit as st


def render_data_exploration(df):
    st.header("Data Exploration")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Fraud Rate", f"{100 * df['label_fraud'].mean():.2f}%")
    c3.metric("Avg Amount", f"${df['amount'].mean():.2f}")

    st.subheader("Fraud by Transaction Type")
    fraud_type = df.groupby("transaction_type")["label_fraud"].mean().reset_index()
    st.plotly_chart(px.bar(fraud_type, x="transaction_type", y="label_fraud", title="Fraud Rate by Transaction Type"), use_container_width=True)

    st.subheader("Amount Distribution")
    st.plotly_chart(px.histogram(df, x="amount", color="label_fraud", nbins=60), use_container_width=True)

    st.subheader("Geo Distance vs IP Risk")
    sampled = df.sample(min(len(df), 4000), random_state=42)
    st.plotly_chart(
        px.scatter(sampled, x="geo_distance_from_home", y="ip_risk_score", color=sampled["label_fraud"].astype(str), opacity=0.5),
        use_container_width=True,
    )
