"""Monitoring utilities (PSI and alerting thresholds)."""

from __future__ import annotations

import numpy as np
import pandas as pd


PSI_COLUMNS = ["feature", "psi", "severity"]


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Calculate PSI for a numeric feature."""
    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)
    if expected.empty or actual.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    breaks = np.unique(np.quantile(expected, quantiles))
    if len(breaks) < 3:
        return 0.0

    expected_bins = pd.cut(expected, bins=breaks, include_lowest=True)
    actual_bins = pd.cut(actual, bins=breaks, include_lowest=True)

    expected_dist = expected_bins.value_counts(normalize=True).sort_index()
    actual_dist = actual_bins.value_counts(normalize=True).sort_index().reindex(expected_dist.index, fill_value=1e-6)

    expected_dist = expected_dist.clip(lower=1e-6)
    actual_dist = actual_dist.clip(lower=1e-6)

    psi = ((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum()
    return float(psi)


def psi_summary(train_df: pd.DataFrame, new_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute PSI table with severity labels."""
    rows = []
    for col in columns:
        if col not in train_df.columns or col not in new_df.columns:
            continue
        psi = population_stability_index(train_df[col], new_df[col])
        if psi < 0.1:
            severity = "stable"
        elif psi < 0.2:
            severity = "moderate"
        else:
            severity = "high"
        rows.append({"feature": col, "psi": psi, "severity": severity})

    if not rows:
        return pd.DataFrame(columns=PSI_COLUMNS)

    return pd.DataFrame(rows, columns=PSI_COLUMNS).sort_values("psi", ascending=False)
