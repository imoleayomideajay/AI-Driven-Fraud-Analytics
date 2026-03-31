"""Feature engineering for fraud analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional fraud-relevant features from raw input."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    out["hour_of_day"] = out["timestamp"].dt.hour.fillna(0).astype(int)
    out["day_of_week"] = out["timestamp"].dt.dayofweek.fillna(0).astype(int)
    out["is_night"] = out["hour_of_day"].isin([0, 1, 2, 3, 4, 23]).astype(int)
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

    out["amount_to_avg_ratio"] = out["amount"] / (out["avg_transaction_amount_7d"] + 1.0)
    out["log_amount"] = np.log1p(out["amount"])
    out["new_beneficiary_flag"] = (out["beneficiary_age_days"] <= 2).astype(int)
    out["account_tenure_risk"] = (out["account_age_days"] < 45).astype(int)
    out["geo_far_flag"] = (out["geo_distance_from_home"] > 300).astype(int)
    out["velocity_high_flag"] = ((out["transaction_count_24h"] >= 8) | (out["velocity_score"] > 0.72)).astype(int)
    out["failed_login_high_flag"] = (out["failed_login_count_24h"] >= 3).astype(int)

    out["risky_device_channel"] = (
        out["device_type"].eq("unknown_device") & out["channel"].isin(["web", "mobile_app"])
    ).astype(int)

    out["timestamp"] = out["timestamp"].astype("string")
    return out


def feature_columns() -> tuple[list[str], list[str]]:
    """Return numerical and categorical feature lists."""
    numeric = [
        "ip_risk_score",
        "geo_distance_from_home",
        "amount",
        "account_age_days",
        "avg_transaction_amount_7d",
        "transaction_count_24h",
        "failed_login_count_24h",
        "beneficiary_age_days",
        "velocity_score",
        "prior_fraud_flag",
        "hour_of_day",
        "day_of_week",
        "is_night",
        "is_weekend",
        "amount_to_avg_ratio",
        "log_amount",
        "new_beneficiary_flag",
        "account_tenure_risk",
        "geo_far_flag",
        "velocity_high_flag",
        "failed_login_high_flag",
        "risky_device_channel",
    ]
    categorical = ["transaction_type", "channel", "merchant_category", "device_type"]
    return numeric, categorical
