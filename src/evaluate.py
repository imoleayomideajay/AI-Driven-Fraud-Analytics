"""Model evaluation utilities for imbalanced fraud detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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


@dataclass
class EvalResult:
    model_name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    false_positive_rate: float
    fraud_capture_top_5pct: float



def select_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray, grid_size: int = 200) -> float:
    """Find classification threshold maximizing F1 score."""
    thresholds = np.linspace(0.05, 0.95, grid_size)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold



def fraud_capture_rate_at_top_n(y_true: np.ndarray, y_prob: np.ndarray, top_pct: float = 0.05) -> float:
    """Recall among top N% highest-risk scored transactions."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).sort_values("y_prob", ascending=False)
    n = max(1, int(len(df) * top_pct))
    top_slice = df.head(n)
    total_fraud = max(df["y_true"].sum(), 1)
    return float(top_slice["y_true"].sum() / total_fraud)



def evaluate_predictions(model_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> EvalResult:
    """Compute fraud-focused binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / max((fp + tn), 1)

    return EvalResult(
        model_name=model_name,
        threshold=threshold,
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_prob),
        pr_auc=average_precision_score(y_true, y_prob),
        false_positive_rate=fpr,
        fraud_capture_top_5pct=fraud_capture_rate_at_top_n(y_true, y_prob, top_pct=0.05),
    )


def eval_result_to_dict(result: EvalResult) -> Dict[str, float | str]:
    """Convert EvalResult to serializable dictionary."""
    return {
        "model_name": result.model_name,
        "threshold": result.threshold,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "roc_auc": result.roc_auc,
        "pr_auc": result.pr_auc,
        "false_positive_rate": result.false_positive_rate,
        "fraud_capture_top_5pct": result.fraud_capture_top_5pct,
    }
