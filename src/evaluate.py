"""Evaluation functions for imbalanced fraud classification."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)



def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fp / max(fp + tn, 1))



def recall_at_top_k(y_true: np.ndarray, y_proba: np.ndarray, k: float = 0.05) -> float:
    """Capture rate among top-k risk transactions."""
    n = len(y_proba)
    top_n = max(1, int(n * k))
    idx = np.argsort(-y_proba)[:top_n]
    return float(y_true[idx].sum() / max(y_true.sum(), 1))



def tune_threshold_for_f1(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """Pick threshold that maximizes F1 on validation/test set."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1



def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "false_positive_rate": false_positive_rate(y_true, y_pred),
        "recall_at_top_5pct": recall_at_top_k(y_true, y_proba, k=0.05),
    }
    return {k: float(v) for k, v in metrics.items()}



def metrics_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(results).T
    return frame.sort_values(["pr_auc", "recall"], ascending=False)
