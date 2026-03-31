"""Monitoring and drift-style checks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_monitoring_summary(scored_df: pd.DataFrame) -> dict:
    total = len(scored_df)
    flagged = int(scored_df["predicted_fraud"].sum()) if "predicted_fraud" in scored_df else 0
    high_risk = int((scored_df.get("risk_band", pd.Series([""] * total)) == "High").sum())
    return {
        "transactions": total,
        "flagged": flagged,
        "flag_rate": flagged / max(total, 1),
        "high_risk": high_risk,
    }



def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Simple PSI drift approximation."""
    expected = expected.astype(float)
    actual = actual.astype(float)
    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles = np.unique(quantiles)
    if len(quantiles) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=quantiles)

    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)



def numeric_drift_report(train_df: pd.DataFrame, new_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    records = []
    for col in columns:
        if col not in train_df.columns or col not in new_df.columns:
            continue
        psi = population_stability_index(train_df[col], new_df[col])
        records.append(
            {
                "feature": col,
                "train_mean": float(train_df[col].mean()),
                "new_mean": float(new_df[col].mean()),
                "psi": psi,
                "drift_flag": "High" if psi >= 0.25 else "Moderate" if psi >= 0.1 else "Low",
            }
        )
    return pd.DataFrame(records).sort_values("psi", ascending=False)
