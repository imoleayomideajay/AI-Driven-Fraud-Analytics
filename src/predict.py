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

OPTIONAL_RULE_COLUMNS = {
    "transaction_count_30m": 0,
    "same_amount_withdrawals_day": 0,
    "dormant_days": 0,
    "pii_change_24h": 0,
    "td_terminated_early": 0,
    "third_party_beneficiary_flag": 0,
    "beneficiary_count_24h": 1,
    "phone_change_24h": 0,
    "mobile_setup_age_days": 365,
    "inflow_24h": 0.0,
    "outflow_1h": 0.0,
}


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


def _is_withdrawal(row: pd.Series) -> bool:
    return row.get("transaction_type", "") in {"cash_withdrawal", "wallet_transfer", "bank_transfer"}


def fraud_rules_panel(row: pd.Series) -> Dict[str, int]:
    """Extended deterministic fraud alert rules requested by operations."""
    hour = pd.to_datetime(row.get("timestamp"), errors="coerce").hour
    hour = int(hour) if not pd.isna(hour) else 12

    amount = float(row.get("amount", 0))
    baseline = float(row.get("avg_transaction_amount_7d", 0)) + 1.0
    geo = float(row.get("geo_distance_from_home", 0))

    rules = {
        "r01_night_txn_23_to_5": int(hour >= 23 or hour <= 5),
        "r02_high_txn_30m": int(row.get("transaction_count_30m", 0) >= 3),
        "r03_same_amount_withdrawals_day": int(_is_withdrawal(row) and row.get("same_amount_withdrawals_day", 0) >= 3),
        "r04_withdrawal_from_dormant_account": int(_is_withdrawal(row) and row.get("dormant_days", 0) >= 180),
        "r05_pii_change_then_withdrawal": int(_is_withdrawal(row) and row.get("pii_change_24h", 0) == 1),
        "r06_td_break_to_third_party": int(row.get("td_terminated_early", 0) == 1 and row.get("third_party_beneficiary_flag", 0) == 1),
        "r07_many_withdrawals_or_transfers": int(_is_withdrawal(row) and row.get("beneficiary_count_24h", 1) >= 4 and row.get("transaction_count_24h", 0) >= 8),
        "r08_phone_change_mobile_setup_then_transfer": int(
            row.get("phone_change_24h", 0) == 1 and row.get("mobile_setup_age_days", 365) <= 3 and row.get("channel", "") == "mobile_app" and _is_withdrawal(row)
        ),
        "r09_pii_change_place_pnd": int(row.get("pii_change_24h", 0) == 1),
        "r10_recent_account_multiple_txn": int(row.get("account_age_days", 9999) <= 30 and row.get("transaction_count_24h", 0) >= 5),
        "r11_new_channel_multiple_txn": int(row.get("mobile_setup_age_days", 365) <= 7 and row.get("transaction_count_24h", 0) >= 5),
        "r12_behavioral_anomaly": int((amount / baseline) >= 3.0 or row.get("transaction_count_24h", 0) >= 10 or geo >= 500),
        "r13_large_inflow_then_outflows": int(row.get("inflow_24h", 0) >= 5000 and row.get("outflow_1h", 0) >= 0.7 * row.get("inflow_24h", 0) and row.get("beneficiary_count_24h", 1) >= 3),
        "r14_location_pattern_anomaly": int(geo >= 700 or (geo >= 400 and row.get("ip_risk_score", 0) >= 70)),
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
    for col, val in OPTIONAL_RULE_COLUMNS.items():
        if col not in out.columns:
            out[col] = val
    return out


def _rules_probability(df: pd.DataFrame) -> np.ndarray:
    """Fallback probability when model artifacts are unavailable."""
    rules_df = df.apply(fraud_rules_panel, axis=1, result_type="expand")
    rule_cols = [c for c in rules_df.columns if c.startswith("r")]
    weights = np.linspace(0.05, 0.11, num=len(rule_cols))
    raw = rules_df[rule_cols].values
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
