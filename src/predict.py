"""Inference utilities for single and batch fraud scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from src.features import add_derived_features
from src.utils import MODELS_DIR, load_json


RISK_BANDS = [(0.0, 0.35, "Low"), (0.35, 0.70, "Medium"), (0.70, 1.01, "High")]


def load_model_artifacts(models_dir: Path = MODELS_DIR):
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


def score_transactions(df: pd.DataFrame, model, threshold: float) -> pd.DataFrame:
    """Score transaction dataframe and append risk outputs."""
    feat = add_derived_features(df)
    X = feat.drop(columns=[c for c in ["label_fraud", "transaction_id", "customer_id", "timestamp"] if c in feat.columns])
    probs = model.predict_proba(X)[:, 1]

    scored = df.copy()
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
        train_mean = float(train_df[col].mean())
        new_mean = float(new_df[col].mean())
        delta_pct = (new_mean - train_mean) / (train_mean + 1e-9)
        rows.append({"feature": col, "train_mean": train_mean, "new_mean": new_mean, "delta_pct": delta_pct})

    return pd.DataFrame(rows).sort_values("delta_pct", key=np.abs, ascending=False)
