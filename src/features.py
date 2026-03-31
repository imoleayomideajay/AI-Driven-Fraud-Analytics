"""Feature engineering for fraud detection."""
from __future__ import annotations

import numpy as np
import pandas as pd



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready fraud-centric features without leakage."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["is_night"] = ((out["hour"] <= 5) | (out["hour"] >= 23)).astype(int)
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

    out["amount_to_avg_ratio_7d"] = out["amount"] / (out["avg_transaction_amount_7d"] + 1)
    out["geo_risk_interaction"] = out["geo_distance_from_home"] * (out["ip_risk_score"] / 100)
    out["velocity_login_interaction"] = out["velocity_score"] * (out["failed_login_count_24h"] + 1)
    out["new_beneficiary_flag"] = (out["beneficiary_age_days"] <= 5).astype(int)
    out["young_account_flag"] = (out["account_age_days"] <= 30).astype(int)

    # Customer behavior gap computed safely from present fields
    out["amount_z_proxy"] = (
        (out["amount"] - out["avg_transaction_amount_7d"]) / np.sqrt(out["avg_transaction_amount_7d"] + 1)
    )

    return out
