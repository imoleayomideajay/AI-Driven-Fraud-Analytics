from __future__ import annotations

import pandas as pd
import streamlit as st

from src.predict import rule_based_alerts, score_dataframe


def render_live_scoring(pipeline, threshold, feature_columns) -> None:
    st.header("Live Transaction Scoring")

    with st.form("single_scoring_form"):
        transaction_type = st.selectbox("Transaction Type", ["card_payment", "bank_transfer", "cash_withdrawal", "bill_payment", "p2p_transfer"])
        channel = st.selectbox("Channel", ["mobile", "web", "atm", "branch", "api"])
        merchant_category = st.selectbox("Merchant Category", ["grocery", "electronics", "gaming", "travel", "utilities", "crypto", "luxury", "food_delivery"])
        device_type = st.selectbox("Device Type", ["known_mobile", "known_web", "new_mobile", "new_web", "emulator"])

        col1, col2, col3 = st.columns(3)
        amount = col1.number_input("Amount", min_value=1.0, value=220.0)
        avg_amt = col2.number_input("Avg Tx Amount 7d", min_value=1.0, value=90.0)
        tx_24h = col3.number_input("Tx Count 24h", min_value=0, value=2)

        col4, col5, col6 = st.columns(3)
        ip_risk = col4.slider("IP Risk Score", 0.0, 100.0, 25.0)
        geo = col5.number_input("Geo Distance From Home", min_value=0.0, value=40.0)
        velocity = col6.slider("Velocity Score", 0.0, 100.0, 18.0)

        col7, col8, col9 = st.columns(3)
        failed_login = col7.number_input("Failed Login Count 24h", min_value=0, value=0)
        bene_age = col8.number_input("Beneficiary Age Days", min_value=0, value=150)
        account_age = col9.number_input("Account Age Days", min_value=1, value=480)

        prior_fraud = st.checkbox("Prior Fraud Flag", value=False)
        submitted = st.form_submit_button("Score Transaction")

    if submitted:
        row = {
            "transaction_id": "TXN_MANUAL_0001",
            "customer_id": "CUST_MANUAL",
            "timestamp": pd.Timestamp.utcnow(),
            "transaction_type": transaction_type,
            "channel": channel,
            "merchant_category": merchant_category,
            "device_type": device_type,
            "ip_risk_score": ip_risk,
            "geo_distance_from_home": geo,
            "amount": amount,
            "account_age_days": account_age,
            "avg_transaction_amount_7d": avg_amt,
            "transaction_count_24h": tx_24h,
            "failed_login_count_24h": failed_login,
            "beneficiary_age_days": bene_age,
            "velocity_score": velocity,
            "prior_fraud_flag": int(prior_fraud),
            "label_fraud": 0,
        }

        input_df = pd.DataFrame([row])
        scored = score_dataframe(input_df, pipeline, threshold, feature_columns)

        probability = float(scored.loc[0, "fraud_probability"])
        band = scored.loc[0, "risk_band"]
        st.metric("Fraud Probability", f"{probability:.2%}")
        st.metric("Risk Band", band)

        st.subheader("Alert Rules Panel")
        alerts = rule_based_alerts(input_df.iloc[0])
        if alerts:
            for a in alerts:
                st.warning(a)
        else:
            st.success("No rule-based alerts triggered")
