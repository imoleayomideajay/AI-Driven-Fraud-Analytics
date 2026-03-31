"""Synthetic transaction generator with realistic fraud patterns."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils import DATA_DIR, SEED, ensure_directories

LOGGER = logging.getLogger(__name__)


TRANSACTION_TYPES = ["card_payment", "bank_transfer", "cash_withdrawal", "bill_payment"]
CHANNELS = ["mobile_app", "web", "atm", "pos_terminal"]
MERCHANT_CATEGORIES = [
    "grocery",
    "utilities",
    "travel",
    "electronics",
    "gambling",
    "crypto",
    "restaurant",
    "retail",
]
DEVICES = ["trusted_mobile", "new_mobile", "desktop", "tablet", "emulator"]



def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))



def generate_synthetic_transactions(n_rows: int = 50_000, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic banking transactions with controlled fraud signals."""
    rng = np.random.default_rng(seed)

    n_customers = max(3_000, int(n_rows / 15))
    customer_ids = [f"CUST_{i:06d}" for i in range(n_customers)]

    start_dt = datetime.utcnow() - timedelta(days=180)
    random_minutes = rng.integers(0, 180 * 24 * 60, size=n_rows)
    timestamps = [start_dt + timedelta(minutes=int(v)) for v in random_minutes]

    transaction_types = rng.choice(TRANSACTION_TYPES, size=n_rows, p=[0.48, 0.30, 0.08, 0.14])
    channels = rng.choice(CHANNELS, size=n_rows, p=[0.45, 0.22, 0.08, 0.25])
    merchant_category = rng.choice(MERCHANT_CATEGORIES, size=n_rows)
    device_type = rng.choice(DEVICES, size=n_rows, p=[0.45, 0.2, 0.2, 0.1, 0.05])

    customer = rng.choice(customer_ids, size=n_rows)

    account_age_days = np.clip(rng.gamma(shape=2.2, scale=220, size=n_rows), 1, None).astype(int)
    beneficiary_age_days = np.clip(rng.gamma(shape=2.0, scale=100, size=n_rows), 0, None).astype(int)
    avg_transaction_amount_7d = np.clip(rng.lognormal(mean=3.7, sigma=0.8, size=n_rows), 5, 4000)
    amount = np.clip(avg_transaction_amount_7d * rng.lognormal(mean=0.05, sigma=0.9, size=n_rows), 1, 25_000)

    ip_risk_score = np.clip(rng.normal(35, 20, n_rows), 1, 100)
    geo_distance_from_home = np.clip(rng.exponential(scale=35, size=n_rows), 0, 8000)
    transaction_count_24h = rng.poisson(lam=2.5, size=n_rows)
    failed_login_count_24h = rng.poisson(lam=0.6, size=n_rows)
    prior_fraud_flag = rng.binomial(1, 0.05, size=n_rows)

    hour = np.array([t.hour for t in timestamps])
    night_flag = ((hour <= 5) | (hour >= 23)).astype(int)

    velocity_score = np.clip(
        0.45 * transaction_count_24h
        + 0.25 * failed_login_count_24h
        + 0.15 * (amount / (avg_transaction_amount_7d + 1))
        + rng.normal(0, 0.7, n_rows),
        0,
        20,
    )

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN_{i:09d}" for i in range(n_rows)],
            "customer_id": customer,
            "timestamp": pd.to_datetime(timestamps),
            "transaction_type": transaction_types,
            "channel": channels,
            "merchant_category": merchant_category,
            "device_type": device_type,
            "ip_risk_score": ip_risk_score,
            "geo_distance_from_home": geo_distance_from_home,
            "amount": amount,
            "account_age_days": account_age_days,
            "avg_transaction_amount_7d": avg_transaction_amount_7d,
            "transaction_count_24h": transaction_count_24h,
            "failed_login_count_24h": failed_login_count_24h,
            "beneficiary_age_days": beneficiary_age_days,
            "velocity_score": velocity_score,
            "prior_fraud_flag": prior_fraud_flag,
        }
    )

    # Pattern injection
    anomalous_amount = (df["amount"] > 3.8 * df["avg_transaction_amount_7d"]).astype(int)
    unusual_geo = (df["geo_distance_from_home"] > 250).astype(int)
    high_velocity = (df["transaction_count_24h"] >= 8).astype(int)
    risky_channel_device = (
        ((df["channel"] == "web") & (df["device_type"].isin(["new_mobile", "emulator"])))
        | ((df["channel"] == "mobile_app") & (df["device_type"] == "emulator"))
    ).astype(int)
    beneficiary_recent = (df["beneficiary_age_days"] <= 3).astype(int)
    risky_type = df["transaction_type"].isin(["bank_transfer", "cash_withdrawal"]).astype(int)
    risky_merchant = df["merchant_category"].isin(["gambling", "crypto"]).astype(int)

    fraud_logit = (
        -5.0
        + 1.00 * high_velocity
        + 0.95 * unusual_geo
        + 1.25 * risky_channel_device
        + 0.90 * anomalous_amount
        + 0.80 * beneficiary_recent
        + 0.75 * risky_type
        + 0.65 * risky_merchant
        + 0.40 * night_flag
        + 0.85 * df["prior_fraud_flag"].values
        + 0.03 * df["ip_risk_score"].values
        + 0.10 * df["failed_login_count_24h"].values
    )

    fraud_prob = _sigmoid(fraud_logit)
    labels = rng.binomial(1, np.clip(fraud_prob, 0.001, 0.95))

    df["label_fraud"] = labels
    return df.sort_values("timestamp").reset_index(drop=True)



def create_and_save_dataset(n_rows: int = 60_000) -> Tuple[pd.DataFrame, str]:
    """Create and save synthetic dataset to /data folder."""
    ensure_directories()
    df = generate_synthetic_transactions(n_rows=n_rows)
    output_path = DATA_DIR / "synthetic_transactions.csv"
    df.to_csv(output_path, index=False)
    fraud_rate = df["label_fraud"].mean()
    LOGGER.info("Saved dataset to %s | rows=%s | fraud_rate=%.3f", output_path, len(df), fraud_rate)
    return df, str(output_path)


if __name__ == "__main__":
    from src.utils import setup_logging

    setup_logging()
    create_and_save_dataset()
