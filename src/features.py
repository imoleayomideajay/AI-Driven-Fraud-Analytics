"""Feature engineering routines for fraud modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical and bucketized time features."""
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_night"] = ((out["hour"] <= 5) | (out["hour"] >= 23)).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


def add_risk_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features known to be useful in fraud analytics."""
    out = df.copy()
    out["amount_vs_avg_ratio"] = out["amount"] / np.maximum(out["avg_transaction_amount_7d"], 1)
    out["geo_velocity_interaction"] = out["geo_distance_from_home"] * (1 + out["transaction_count_24h"])
    out["login_velocity_interaction"] = out["failed_login_count_24h"] * out["transaction_count_24h"]
    out["risk_composite"] = (
        0.35 * out["ip_risk_score"]
        + 0.25 * out["velocity_score"]
        + 0.20 * np.clip(out["amount_vs_avg_ratio"] * 10, 0, 100)
        + 0.20 * np.clip(out["geo_distance_from_home"] / 10, 0, 100)
    )
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full feature engineering pipeline."""
    return add_risk_interactions(add_time_features(df))
