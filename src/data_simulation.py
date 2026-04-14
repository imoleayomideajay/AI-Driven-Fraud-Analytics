"""Synthetic transaction simulator with realistic fraud behavior."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DATA_DIR, RANDOM_SEED, seeded_rng

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    n_rows: int = 50_000
    n_customers: int = 7_500
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    fraud_base_rate: float = 0.03
    random_seed: int = RANDOM_SEED


TRANSACTION_TYPES = ["card_payment", "bank_transfer", "cash_withdrawal", "wallet_transfer"]
CHANNELS = ["mobile_app", "web", "atm", "pos"]
MERCHANT_CATEGORIES = [
    "grocery",
    "electronics",
    "travel",
    "gaming",
    "utilities",
    "fin_services",
    "crypto",
    "cash_services",
]
DEVICES = ["ios", "android", "desktop", "unknown_device"]


def _random_timestamps(rng: np.random.Generator, n_rows: int, start_date: str, end_date: str) -> pd.Series:
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    seconds = int((end - start).total_seconds())
    sampled = rng.integers(0, seconds, size=n_rows)
    return pd.Series([start + timedelta(seconds=int(v)) for v in sampled])


def generate_synthetic_transactions(config: SimulationConfig = SimulationConfig()) -> pd.DataFrame:
    """Generate synthetic transaction data with embedded fraud patterns."""
    rng = seeded_rng(config.random_seed)
    n = config.n_rows

    customer_ids = np.array([f"CUST_{i:06d}" for i in range(config.n_customers)])
    tx_customer_ids = rng.choice(customer_ids, size=n, replace=True)

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN_{i:09d}" for i in range(n)],
            "customer_id": tx_customer_ids,
            "timestamp": _random_timestamps(rng, n, config.start_date, config.end_date),
            "transaction_type": rng.choice(TRANSACTION_TYPES, n, p=[0.45, 0.25, 0.10, 0.20]),
            "channel": rng.choice(CHANNELS, n, p=[0.44, 0.24, 0.08, 0.24]),
            "merchant_category": rng.choice(MERCHANT_CATEGORIES, n),
            "device_type": rng.choice(DEVICES, n, p=[0.30, 0.35, 0.30, 0.05]),
            "ip_risk_score": np.clip(rng.beta(2, 5, n) * 100, 0, 100),
            "geo_distance_from_home": np.clip(rng.gamma(shape=2.0, scale=14.0, size=n), 0, 2500),
            "amount": np.clip(rng.lognormal(mean=3.7, sigma=0.9, size=n), 1.0, 25000),
            "account_age_days": np.clip(rng.gamma(5.0, 180, n).astype(int), 1, 3650),
            "avg_transaction_amount_7d": np.clip(rng.lognormal(mean=3.5, sigma=0.6, size=n), 5, 3500),
            "transaction_count_24h": np.clip(rng.poisson(2.2, n), 0, 60),
            "failed_login_count_24h": np.clip(rng.poisson(0.35, n), 0, 20),
            "beneficiary_age_days": np.clip(rng.gamma(3.0, 120, n).astype(int), 0, 3000),
            "velocity_score": np.clip(rng.normal(0.28, 0.16, n), 0, 1),
            "prior_fraud_flag": rng.binomial(1, 0.07, n),            "transaction_count_30m": np.clip(rng.poisson(0.8, n), 0, 20),
            "same_amount_withdrawals_day": np.clip(rng.poisson(0.5, n), 0, 12),
            "dormant_days": np.clip(rng.gamma(1.5, 40, n).astype(int), 0, 1200),
            "pii_change_24h": rng.binomial(1, 0.02, n),
            "td_terminated_early": rng.binomial(1, 0.01, n),
            "third_party_beneficiary_flag": rng.binomial(1, 0.08, n),
            "beneficiary_count_24h": np.clip(rng.poisson(1.2, n), 1, 25),
            "phone_change_24h": rng.binomial(1, 0.01, n),
            "mobile_setup_age_days": np.clip(rng.gamma(2.0, 80, n).astype(int), 0, 2000),
            "inflow_24h": np.clip(rng.lognormal(mean=4.4, sigma=1.0, size=n), 0, 150000),
            "outflow_1h": np.clip(rng.lognormal(mean=3.8, sigma=1.1, size=n), 0, 100000),
        }
    )

    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)

    amount_ratio = df["amount"] / (df["avg_transaction_amount_7d"] + 1)
    risky_combo = ((df["channel"].isin(["web", "mobile_app"])) & (df["device_type"] == "unknown_device")).astype(int)
    geo_anomaly = (df["geo_distance_from_home"] > 300).astype(int)
    high_velocity = ((df["transaction_count_24h"] >= 8) | (df["velocity_score"] > 0.72)).astype(int)
    new_beneficiary = (df["beneficiary_age_days"] <= 2).astype(int)
    type_risk = df["transaction_type"].map(
        {"card_payment": 0.4, "bank_transfer": 1.2, "cash_withdrawal": 0.9, "wallet_transfer": 0.7}
    )
    merchant_risk = df["merchant_category"].map(
        {"crypto": 1.5, "gaming": 0.9, "cash_services": 1.2, "travel": 0.6}
    ).fillna(0.3)

    linear_risk = (
        -4.8
        + 0.024 * df["ip_risk_score"]
        + 1.10 * high_velocity
        + 0.95 * geo_anomaly
        + 1.15 * risky_combo
        + 0.92 * new_beneficiary
        + 0.20 * df["failed_login_count_24h"]
        + 0.90 * df["is_night"]
        + 0.45 * df["prior_fraud_flag"]
        + 0.8 * np.log1p(amount_ratio)
        + 0.55 * type_risk
        + 0.55 * (df["transaction_count_30m"] >= 3).astype(int)
        + 0.70 * (df["same_amount_withdrawals_day"] >= 3).astype(int)
        + 0.65 * ((df["dormant_days"] >= 180) & (df["transaction_type"].isin(["cash_withdrawal", "bank_transfer"]))).astype(int)
        + 0.80 * ((df["pii_change_24h"] == 1) & (df["transaction_type"].isin(["cash_withdrawal", "bank_transfer", "wallet_transfer"]))).astype(int)
        + 0.95 * ((df["td_terminated_early"] == 1) & (df["third_party_beneficiary_flag"] == 1)).astype(int)
        + 0.75 * ((df["phone_change_24h"] == 1) & (df["mobile_setup_age_days"] <= 3) & (df["channel"] == "mobile_app")).astype(int)
        + 0.60 * ((df["account_age_days"] <= 30) & (df["transaction_count_24h"] >= 5)).astype(int)
        + 0.70 * ((df["mobile_setup_age_days"] <= 7) & (df["transaction_count_24h"] >= 5)).astype(int)
        + 0.90 * ((df["inflow_24h"] >= 5000) & (df["outflow_1h"] >= 0.7 * df["inflow_24h"]) & (df["beneficiary_count_24h"] >= 3)).astype(int)
        + 0.65 * ((df["geo_distance_from_home"] >= 700) | ((df["geo_distance_from_home"] >= 400) & (df["ip_risk_score"] >= 70))).astype(int)
    )

    calibrated_shift = np.log(config.fraud_base_rate / (1 - config.fraud_base_rate)) - np.mean(linear_risk)
    fraud_prob = 1 / (1 + np.exp(-(linear_risk + calibrated_shift)))
    df["label_fraud"] = rng.binomial(1, np.clip(fraud_prob, 0.001, 0.97))

    df = df.drop(columns=["hour", "is_night"])
    logger.info("Generated synthetic dataset shape=%s fraud_rate=%.4f", df.shape, df["label_fraud"].mean())
    return df


def save_dataset(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Save generated data and return output path."""
    path = output_path or (DATA_DIR / "synthetic_transactions.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved synthetic data to %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = generate_synthetic_transactions()
    save_dataset(data)
