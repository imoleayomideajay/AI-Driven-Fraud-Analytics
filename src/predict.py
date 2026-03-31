"""Prediction and fraud alert rules utilities."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.features import engineer_features
from src.utils import MODELS_DIR, load_artifact



def load_model_assets() -> Dict[str, object]:
    pipeline = load_artifact(MODELS_DIR / "champion_pipeline.joblib")
    metadata = load_artifact(MODELS_DIR / "metadata.joblib")
    return {"pipeline": pipeline, "metadata": metadata}



def score_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    assets = load_model_assets()
    pipeline = assets["pipeline"]
    threshold = float(assets["metadata"]["threshold"])

    fe_df = engineer_features(raw_df)
    feature_cols = assets["metadata"]["feature_columns"]
    X = fe_df[feature_cols]

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        scores = pipeline.predict_proba(X)[:, 1]
    else:
        raw = -pipeline.decision_function(X)
        scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    out = raw_df.copy()
    out["fraud_score"] = scores
    out["fraud_pred"] = (scores >= threshold).astype(int)
    out["risk_label"] = pd.cut(
        scores,
        bins=[-0.01, 0.35, 0.70, 1.0],
        labels=["Low", "Medium", "High"],
    )
    return out



def generate_rule_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Generate transparent business rules alongside model score."""
    alerts: List[List[str]] = []
    for _, row in df.iterrows():
        tx_alerts: List[str] = []
        if row.get("transaction_count_24h", 0) >= 8:
            tx_alerts.append("High velocity")
        if row.get("geo_distance_from_home", 0) > 250:
            tx_alerts.append("Unusual geo distance")
        if row.get("failed_login_count_24h", 0) >= 3:
            tx_alerts.append("Multiple failed logins")
        if row.get("amount", 0) > 3 * (row.get("avg_transaction_amount_7d", 1) + 1):
            tx_alerts.append("Amount anomaly vs 7d history")
        if row.get("beneficiary_age_days", 999) <= 3:
            tx_alerts.append("New beneficiary")
        if row.get("device_type") in ["emulator", "new_mobile"] and row.get("channel") == "web":
            tx_alerts.append("Risky device/channel pair")
        alerts.append(tx_alerts)

    out = df.copy()
    out["rule_alert_count"] = [len(a) for a in alerts]
    out["rule_alerts"] = [", ".join(a) if a else "None" for a in alerts]
    return out
