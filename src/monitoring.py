"""Monitoring and drift summary helpers."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd



def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index implementation."""
    expected = pd.to_numeric(expected, errors="coerce").fillna(0)
    actual = pd.to_numeric(actual, errors="coerce").fillna(0)

    qs = np.linspace(0, 1, bins + 1)
    breaks = np.unique(np.quantile(expected, qs))
    if len(breaks) < 3:
        return 0.0

    e_counts, _ = np.histogram(expected, bins=breaks)
    a_counts, _ = np.histogram(actual, bins=breaks)

    e_perc = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, None)
    a_perc = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))



def drift_summary(train_df: pd.DataFrame, new_df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        if col in train_df.columns and col in new_df.columns:
            score = psi(train_df[col], new_df[col])
            status = "Stable" if score < 0.1 else "Moderate" if score < 0.25 else "Significant"
            rows.append({"feature": col, "psi": score, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)



def monitoring_kpis(scored_df: pd.DataFrame) -> Dict[str, float]:
    fraud_rate = float((scored_df["fraud_pred"] == 1).mean())
    high_risk_rate = float((scored_df["risk_label"].astype(str) == "High").mean())
    avg_score = float(scored_df["fraud_score"].mean())
    alerts_per_tx = float(scored_df["rule_alert_count"].mean()) if "rule_alert_count" in scored_df else 0.0
    return {
        "fraud_rate": fraud_rate,
        "high_risk_rate": high_risk_rate,
        "avg_score": avg_score,
        "alerts_per_tx": alerts_per_tx,
    }
