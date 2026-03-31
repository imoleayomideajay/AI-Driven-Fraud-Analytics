"""Prediction and scoring utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from src.features import build_features



def load_artifacts(models_dir: Path) -> Tuple[object, float, list[str]]:
    pipeline = joblib.load(models_dir / "champion_pipeline.joblib")
    threshold = float(joblib.load(models_dir / "champion_threshold.joblib"))
    feature_columns = joblib.load(models_dir / "feature_columns.joblib")
    return pipeline, threshold, feature_columns



def risk_band(probability: float) -> str:
    if probability >= 0.80:
        return "High"
    if probability >= 0.45:
        return "Medium"
    return "Low"



def rule_based_alerts(row: pd.Series) -> list[str]:
    alerts: list[str] = []
    if row.get("transaction_count_24h", 0) >= 10:
        alerts.append("High velocity in last 24h")
    if row.get("geo_distance_from_home", 0) >= 300:
        alerts.append("Unusual geo-distance")
    if row.get("amount", 0) > 3.5 * max(row.get("avg_transaction_amount_7d", 1), 1):
        alerts.append("Amount anomalous vs 7-day average")
    if row.get("failed_login_count_24h", 0) >= 3:
        alerts.append("Multiple recent failed logins")
    if row.get("beneficiary_age_days", 9999) <= 7:
        alerts.append("Recently added beneficiary")
    if str(row.get("device_type", "")) in {"emulator", "new_web"} and str(row.get("channel", "")) in {"web", "api"}:
        alerts.append("Risky device/channel pattern")
    return alerts



def score_dataframe(df: pd.DataFrame, pipeline: object, threshold: float, feature_columns: list[str]) -> pd.DataFrame:
    featured = build_features(df.copy())
    X = featured.drop(columns=["label_fraud", "transaction_id", "customer_id", "timestamp"], errors="ignore")
    X = X.reindex(columns=feature_columns)
    probs = pipeline.predict_proba(X)[:, 1]
    labels = (probs >= threshold).astype(int)

    out = df.copy()
    out["fraud_probability"] = probs
    out["predicted_fraud"] = labels
    out["risk_band"] = [risk_band(p) for p in probs]
    out["alert_rules"] = out.apply(lambda row: "; ".join(rule_based_alerts(row)) or "No rule-based alert", axis=1)
    return out
