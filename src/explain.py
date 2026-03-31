"""Explainability helpers with SHAP fallback logic."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def global_feature_importance(pipeline: Pipeline, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Get global feature importance using model-native methods."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_t = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        importance = np.std(X_t, axis=0).A1 if hasattr(X_t, "A1") else np.std(X_t, axis=0)

    out = pd.DataFrame({"feature": feature_names, "importance": importance})
    return out.sort_values("importance", ascending=False)


def shap_values_if_available(pipeline: Pipeline, X_sample: pd.DataFrame) -> Dict[str, object]:
    """Return SHAP values where feasible, else fallback details."""
    try:
        import shap

        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        transformed = preprocessor.transform(X_sample)
        explainer = shap.Explainer(model, transformed)
        shap_values = explainer(transformed)
        return {"available": True, "values": shap_values}
    except Exception as exc:
        logger.warning("SHAP unavailable or failed: %s", exc)
        return {"available": False, "reason": str(exc)}
