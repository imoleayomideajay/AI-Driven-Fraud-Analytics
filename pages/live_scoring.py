from __future__ import annotations

import pandas as pd
import streamlit as st

from src.predict import score_transactions


def _build_single_input() -> pd.DataFrame:
    st.subheader("Single Transaction Input")
    amount = st.number_input("Amount", min_value=1.0, value=120.0)
    tx_type = st.selectbox("Transaction Type", ["card_payment", "bank_transfer", "cash_withdrawal", "wallet_transfer"])
    channel = st.selectbox("Channel", ["mobile_app", "web", "atm", "pos"])
    merchant = st.selectbox("Merchant Category", ["grocery", "electronics", "travel", "gaming", "utilities", "fin_services", "crypto", "cash_services"])
    device = st.selectbox("Device Type", ["ios", "android", "desktop", "unknown_device"])
    ip_risk = st.slider("IP Risk Score", 0.0, 100.0, 45.0)
    geo = st.number_input("Geo Distance from Home (km)", min_value=0.0, value=30.0)
    account_age = st.number_input("Account Age Days", min_value=1, value=720)
    avg_7d = st.number_input("Avg Transaction Amount 7d", min_value=1.0, value=90.0)
    tx_24h = st.number_input("Transaction Count 24h", min_value=0, value=2)
    fail_login = st.number_input("Failed Login Count 24h", min_value=0, value=0)
    bene_age = st.number_input("Beneficiary Age Days", min_value=0, value=240)
    velocity = st.slider("Velocity Score", 0.0, 1.0, 0.2)
    prior_fraud = st.selectbox("Prior Fraud Flag", [0, 1])

    return pd.DataFrame(
        [
            {
                "transaction_id": "TXN_LIVE_0001",
                "customer_id": "CUST_LIVE_0001",
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "transaction_type": tx_type,
                "channel": channel,
                "merchant_category": merchant,
                "device_type": device,
                "ip_risk_score": ip_risk,
                "geo_distance_from_home": geo,
                "amount": amount,
                "account_age_days": account_age,
                "avg_transaction_amount_7d": avg_7d,
                "transaction_count_24h": tx_24h,
                "failed_login_count_24h": fail_login,
                "beneficiary_age_days": bene_age,
                "velocity_score": velocity,
                "prior_fraud_flag": prior_fraud,
            }
        ]
    )


def render_live_scoring(model, threshold: float) -> None:
    st.header("Live Scoring")
    sample = _build_single_input()

    if st.button("Score Transaction"):
        scored = score_transactions(sample, model, threshold)
        record = scored.iloc[0]
        st.metric("Fraud Probability", f"{record['fraud_probability']:.2%}")
        st.metric("Risk Label", record["risk_label"])
        st.metric("ML Prediction", "Fraud" if record["fraud_prediction"] == 1 else "Legit")

        st.subheader("Alert Rules Panel")
        rule_cols = [
            "high_ip_risk",
            "far_from_home",
            "high_velocity",
            "new_beneficiary",
            "high_failed_logins",
            "anomalous_amount",
            "rules_triggered",
        ]
        st.dataframe(scored[rule_cols], use_container_width=True)
