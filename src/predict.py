"""Inference utilities for single and batch fraud scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.features import add_derived_features
from src.utils import MODELS_DIR, load_json


RISK_BANDS = [(0.0, 0.35, "Low"), (0.35, 0.70, "Medium"), (0.70, 1.01, "High")]
REQUIRED_INPUT_COLUMNS = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "transaction_type",
    "channel",
    "merchant_category",
    "device_type",
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
]


def load_model_artifacts(models_dir: Path = MODELS_DIR) -> Tuple[object, dict]:
    """Load champion model and metadata from disk."""
    model = joblib.load(models_dir / "champion_model.joblib")
    metadata = load_json(models_dir / "training_metadata.json")
    return model, metadata


def risk_label(score: float) -> str:
    """Map probability score to human-readable risk label."""
    for lo, hi, label in RISK_BANDS:
        if lo <= score < hi:
            return label
    return "High"


def fraud_rules_panel(row: pd.Series) -> Dict[str, int]:
    """Simple deterministic fraud alert rules to complement ML score."""
    rules = {
        "high_ip_risk": int(row.get("ip_risk_score", 0) >= 75),
        "far_from_home": int(row.get("geo_distance_from_home", 0) > 500),
        "high_velocity": int(row.get("transaction_count_24h", 0) >= 10 or row.get("velocity_score", 0) > 0.75),
        "new_beneficiary": int(row.get("beneficiary_age_days", 9999) <= 2),
        "high_failed_logins": int(row.get("failed_login_count_24h", 0) >= 3),
        "anomalous_amount": int(row.get("amount", 0) > 3.0 * (row.get("avg_transaction_amount_7d", 0) + 1)),
    }
    rules["rules_triggered"] = int(sum(rules.values()))
    return rules


def _validate_and_fill_input(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure input has expected columns with safe defaults for scoring."""
    out = df.copy()
    defaults = {
        "transaction_id": "TXN_MISSING",
        "customer_id": "CUST_MISSING",
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "transaction_type": "card_payment",
        "channel": "mobile_app",
        "merchant_category": "grocery",
        "device_type": "android",
        "ip_risk_score": 25.0,
        "geo_distance_from_home": 20.0,
        "amount": 50.0,
        "account_age_days": 365,
        "avg_transaction_amount_7d": 55.0,
        "transaction_count_24h": 1,
        "failed_login_count_24h": 0,
        "beneficiary_age_days": 180,
        "velocity_score": 0.2,
        "prior_fraud_flag": 0,
    }
    for col in REQUIRED_INPUT_COLUMNS:
        if col not in out.columns:
            out[col] = defaults[col]
    return out[REQUIRED_INPUT_COLUMNS + [c for c in out.columns if c not in REQUIRED_INPUT_COLUMNS]]


def _rules_probability(df: pd.DataFrame) -> np.ndarray:
    """Fallback probability when model artifacts are unavailable."""
    rules_df = df.apply(fraud_rules_panel, axis=1, result_type="expand")
    weights = np.array([0.18, 0.20, 0.22, 0.15, 0.10, 0.15])
    raw = rules_df[["high_ip_risk", "far_from_home", "high_velocity", "new_beneficiary", "high_failed_logins", "anomalous_amount"]].values
    probs = np.clip(raw @ weights, 0, 0.98)
    return probs


def score_transactions(df: pd.DataFrame, model, threshold: float) -> pd.DataFrame:
    """Score transaction dataframe and append risk outputs."""
    safe_df = _validate_and_fill_input(df)

    if model is None:
        probs = _rules_probability(safe_df)
    else:
        feat = add_derived_features(safe_df)
        X = feat.drop(columns=[c for c in ["label_fraud", "transaction_id", "customer_id", "timestamp"] if c in feat.columns])
        probs = model.predict_proba(X)[:, 1]

    scored = safe_df.copy()
    scored["fraud_probability"] = probs
    scored["fraud_prediction"] = (probs >= threshold).astype(int)
    scored["risk_label"] = scored["fraud_probability"].apply(risk_label)

    rules_df = scored.apply(fraud_rules_panel, axis=1, result_type="expand")
    scored = pd.concat([scored, rules_df], axis=1)
    return scored


def simple_drift_check(train_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight drift check comparing means and category proportions."""
    numeric_cols = [
        "amount",
        "ip_risk_score",
        "geo_distance_from_home",
        "transaction_count_24h",
        "velocity_score",
    ]
    rows = []
    for col in numeric_cols:
        if col not in train_df.columns or col not in new_df.columns:
            continue
        train_mean = float(train_df[col].mean())
        new_mean = float(new_df[col].mean())
        delta_pct = (new_mean - train_mean) / (train_mean + 1e-9)
        rows.append({"feature": col, "train_mean": train_mean, "new_mean": new_mean, "delta_pct": delta_pct})

    if not rows:
        return pd.DataFrame(columns=["feature", "train_mean", "new_mean", "delta_pct"])
    return pd.DataFrame(rows).sort_values("delta_pct", key=np.abs, ascending=False)
