from __future__ import annotations

import pandas as pd
import streamlit as st

from src.explain import explain_single_prediction
from src.predict import generate_rule_alerts, score_transactions



def render() -> None:
    st.header("Live Scoring (Single Transaction)")

    with st.form("live_form"):
        amount = st.number_input("Amount", min_value=1.0, value=120.0)
        avg7 = st.number_input("Avg transaction amount 7d", min_value=1.0, value=95.0)
        geo = st.number_input("Geo distance from home", min_value=0.0, value=12.0)
        ip_risk = st.slider("IP risk score", 1, 100, 32)
        tx_count = st.slider("Transaction count 24h", 0, 30, 2)
        fail_logins = st.slider("Failed login count 24h", 0, 10, 0)
        beneficiary_age = st.number_input("Beneficiary age days", min_value=0, value=45)
        velocity = st.number_input("Velocity score", min_value=0.0, value=3.2)
        acct_age = st.number_input("Account age days", min_value=1, value=300)
        prior_fraud = st.selectbox("Prior fraud flag", [0, 1], index=0)
        transaction_type = st.selectbox("Transaction type", ["card_payment", "bank_transfer", "cash_withdrawal", "bill_payment"])
        channel = st.selectbox("Channel", ["mobile_app", "web", "atm", "pos_terminal"])
        merchant_category = st.selectbox("Merchant category", ["grocery", "utilities", "travel", "electronics", "gambling", "crypto", "restaurant", "retail"])
        device_type = st.selectbox("Device type", ["trusted_mobile", "new_mobile", "desktop", "tablet", "emulator"])
        submitted = st.form_submit_button("Score Transaction")

    if submitted:
        tx = pd.DataFrame(
            [
                {
                    "transaction_id": "TXN_LIVE_0001",
                    "customer_id": "CUST_LIVE",
                    "timestamp": pd.Timestamp.utcnow(),
                    "transaction_type": transaction_type,
                    "channel": channel,
                    "merchant_category": merchant_category,
                    "device_type": device_type,
                    "ip_risk_score": ip_risk,
                    "geo_distance_from_home": geo,
                    "amount": amount,
                    "account_age_days": acct_age,
                    "avg_transaction_amount_7d": avg7,
                    "transaction_count_24h": tx_count,
                    "failed_login_count_24h": fail_logins,
                    "beneficiary_age_days": beneficiary_age,
                    "velocity_score": velocity,
                    "prior_fraud_flag": prior_fraud,
                }
            ]
        )
        scored = score_transactions(tx)
        scored = generate_rule_alerts(scored)
        explanation = explain_single_prediction(tx)

        st.subheader("Scoring Result")
        st.write(scored[["fraud_score", "fraud_pred", "risk_label", "rule_alert_count", "rule_alerts"]])
        st.subheader("Top Prediction Drivers")
        st.json(explanation)
