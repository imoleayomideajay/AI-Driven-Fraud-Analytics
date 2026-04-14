"""Explainability helpers with SHAP fallback logic."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features import add_derived_features

logger = logging.getLogger(__name__)


def _prepare_explainability_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare input frame so it matches model training feature schema."""
    feat = add_derived_features(df)
    drop_cols = [c for c in ["label_fraud", "transaction_id", "customer_id", "timestamp"] if c in feat.columns]
    return feat.drop(columns=drop_cols)


def global_feature_importance(pipeline: Pipeline, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Get global feature importance using model-native methods."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    prepared = _prepare_explainability_frame(X_sample)
    X_t = preprocessor.transform(prepared)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        dense = X_t.toarray() if hasattr(X_t, "toarray") else np.asarray(X_t)
        importance = np.std(dense, axis=0)

    out = pd.DataFrame({"feature": feature_names, "importance": importance})
    return out.sort_values("importance", ascending=False)


def shap_values_if_available(pipeline: Pipeline, X_sample: pd.DataFrame) -> Dict[str, object]:
    """Return SHAP values where feasible, else fallback details."""
    try:
        import shap

        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        prepared = _prepare_explainability_frame(X_sample)
        transformed = preprocessor.transform(prepared)
        explainer = shap.Explainer(model, transformed)
        shap_values = explainer(transformed)
        return {"available": True, "values": shap_values}
    except Exception as exc:
        logger.warning("SHAP unavailable or failed: %s", exc)
        return {"available": False, "reason": str(exc)}
