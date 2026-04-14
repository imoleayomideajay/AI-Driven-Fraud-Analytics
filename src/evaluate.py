"""Model evaluation helpers for imbalanced fraud detection."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute fraud-focused binary classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn + 1e-9)

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "false_positive_rate": fpr,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def capture_rate_at_top_n(y_true: np.ndarray, y_prob: np.ndarray, top_n_pct: float = 0.05) -> float:
    """Return fraud recall captured in top N% highest risk transactions."""
    if not 0 < top_n_pct <= 1:
        raise ValueError("top_n_pct must be in (0, 1].")
    n = len(y_true)
    cutoff = max(1, int(n * top_n_pct))
    ranking = np.argsort(-y_prob)
    top_idx = ranking[:cutoff]
    total_fraud = np.sum(y_true)
    if total_fraud == 0:
        return 0.0
    return float(np.sum(y_true[top_idx]) / total_fraud)


def tune_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Scan thresholds and return best threshold and corresponding F1."""
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])


def tune_threshold_with_precision_floor(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.35,
) -> Tuple[float, Dict[str, float]]:
    """Choose highest recall threshold that meets minimum precision; fallback to max F1 threshold."""
    thresholds = np.linspace(0.05, 0.95, 91)
    candidates = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        candidates.append((t, p, r, f1))

    valid = [row for row in candidates if row[1] >= min_precision]
    if valid:
        best = max(valid, key=lambda row: (row[2], row[3]))
    else:
        best = max(candidates, key=lambda row: row[3])

    t, p, r, f1 = best
    return float(t), {"precision": float(p), "recall": float(r), "f1": float(f1)}


def summarize_cv_results(cv_scores: dict) -> pd.DataFrame:
    """Convert CV score dict to tidy dataframe."""
    rows = []
    for metric_name, values in cv_scores.items():
        rows.append(
            {
                "metric": metric_name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        )
    return pd.DataFrame(rows)
