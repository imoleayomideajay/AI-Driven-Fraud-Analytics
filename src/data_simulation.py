"""Synthetic fraud transaction data generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    n_rows: int = 60_000
    n_customers: int = 8_000
    seed: int = 42


def _sample_categorical(rng: np.random.Generator, values: list[str], probs: list[float], size: int) -> np.ndarray:
    return rng.choice(values, p=probs, size=size)


def generate_synthetic_transactions(config: SimulationConfig) -> pd.DataFrame:
    """Generate synthetic banking transactions with realistic fraud signals."""
    rng = np.random.default_rng(config.seed)

    customer_ids = np.array([f"CUST_{idx:06d}" for idx in range(config.n_customers)])
    tx_customer = rng.choice(customer_ids, size=config.n_rows, replace=True)

    end_time = pd.Timestamp.utcnow().floor("min")
    start_time = end_time - pd.Timedelta(days=120)
    timestamps = pd.to_datetime(rng.integers(start_time.value // 10**9, end_time.value // 10**9, size=config.n_rows), unit="s")

    transaction_type = _sample_categorical(
        rng,
        ["card_payment", "bank_transfer", "cash_withdrawal", "bill_payment", "p2p_transfer"],
        [0.42, 0.20, 0.11, 0.14, 0.13],
        config.n_rows,
    )
    channel = _sample_categorical(rng, ["mobile", "web", "atm", "branch", "api"], [0.45, 0.27, 0.12, 0.06, 0.10], config.n_rows)
    merchant_category = _sample_categorical(
        rng,
        ["grocery", "electronics", "gaming", "travel", "utilities", "crypto", "luxury", "food_delivery"],
        [0.25, 0.12, 0.10, 0.09, 0.16, 0.05, 0.08, 0.15],
        config.n_rows,
    )
    device_type = _sample_categorical(rng, ["known_mobile", "known_web", "new_mobile", "new_web", "emulator"], [0.34, 0.26, 0.16, 0.18, 0.06], config.n_rows)

    # Numeric features
    account_age_days = rng.integers(30, 3650, size=config.n_rows)
    avg_transaction_amount_7d = np.clip(rng.normal(85, 40, size=config.n_rows), 5, None)

    amount_multiplier = rng.lognormal(mean=0.0, sigma=0.75, size=config.n_rows)
    amount = np.clip(avg_transaction_amount_7d * amount_multiplier, 1, 8000)

    transaction_count_24h = np.clip(rng.poisson(lam=2.2, size=config.n_rows), 0, 30)
    failed_login_count_24h = np.clip(rng.poisson(lam=0.35, size=config.n_rows), 0, 12)
    geo_distance_from_home = np.clip(rng.gamma(2.0, 18.0, size=config.n_rows), 0, 10000)
    ip_risk_score = np.clip(rng.beta(1.8, 5.8, size=config.n_rows) * 100, 0, 100)
    beneficiary_age_days = rng.integers(0, 2200, size=config.n_rows)
    velocity_score = np.clip(transaction_count_24h * 6 + failed_login_count_24h * 5 + rng.normal(10, 7, config.n_rows), 0, 100)
    prior_fraud_flag = rng.binomial(1, 0.05, size=config.n_rows)

    hour_of_day = pd.Series(timestamps).dt.hour.to_numpy()
    night_activity = ((hour_of_day <= 5) | (hour_of_day >= 23)).astype(int)

    # Realistic fraud patterns combined into a probabilistic score
    anomalous_amount = (amount / np.maximum(avg_transaction_amount_7d, 1.0)) > 3.8
    high_velocity = transaction_count_24h >= 10
    unusual_geo = geo_distance_from_home >= 350
    risky_device_channel = np.isin(device_type, ["emulator", "new_web"]) & np.isin(channel, ["web", "api"])
    rapid_beneficiary = beneficiary_age_days <= 5
    risky_tx_type = np.isin(transaction_type, ["bank_transfer", "p2p_transfer", "cash_withdrawal"])
    risky_merchant = np.isin(merchant_category, ["crypto", "gaming", "luxury"])

    logit = (
        -6.2
        + 0.03 * ip_risk_score
        + 0.012 * velocity_score
        + 0.0014 * geo_distance_from_home
        + 0.6 * prior_fraud_flag
        + 0.42 * failed_login_count_24h
        + 1.4 * anomalous_amount.astype(float)
        + 1.2 * high_velocity.astype(float)
        + 1.0 * unusual_geo.astype(float)
        + 1.4 * risky_device_channel.astype(float)
        + 0.9 * rapid_beneficiary.astype(float)
        + 0.5 * risky_tx_type.astype(float)
        + 0.35 * risky_merchant.astype(float)
        + 0.5 * night_activity
    )
    fraud_probability = 1 / (1 + np.exp(-logit))
    label_fraud = rng.binomial(1, np.clip(fraud_probability, 0, 0.95))

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN_{i:09d}" for i in range(config.n_rows)],
            "customer_id": tx_customer,
            "timestamp": timestamps,
            "transaction_type": transaction_type,
            "channel": channel,
            "merchant_category": merchant_category,
            "device_type": device_type,
            "ip_risk_score": ip_risk_score.round(2),
            "geo_distance_from_home": geo_distance_from_home.round(2),
            "amount": amount.round(2),
            "account_age_days": account_age_days,
            "avg_transaction_amount_7d": avg_transaction_amount_7d.round(2),
            "transaction_count_24h": transaction_count_24h,
            "failed_login_count_24h": failed_login_count_24h,
            "beneficiary_age_days": beneficiary_age_days,
            "velocity_score": velocity_score.round(2),
            "prior_fraud_flag": prior_fraud_flag,
            "label_fraud": label_fraud,
        }
    ).sort_values("timestamp").reset_index(drop=True)

    LOGGER.info("Synthetic dataset generated: %s rows, fraud rate %.2f%%", len(df), 100 * df["label_fraud"].mean())
    return df


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Save generated dataset to CSV."""
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved synthetic data to %s", output_path)


if __name__ == "__main__":
    from src.utils import ensure_directories, setup_logging

    setup_logging()
    ensure_directories(["data"])
    dataset = generate_synthetic_transactions(SimulationConfig())
    save_dataset(dataset, "data/synthetic_transactions.csv")
