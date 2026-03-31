"""Explainability helpers with SHAP + fallback support."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.features import build_features



def global_feature_importance(pipeline: object, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Return model-driven global feature importance.

    Uses tree importances when available, otherwise absolute model coefficients.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(X_sample)

    feature_names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.asarray(transformed).std(axis=0)

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)



def shap_values_for_row(pipeline: object, X: pd.DataFrame, row_idx: int = 0) -> Optional[pd.DataFrame]:
    """Compute SHAP values for one row when SHAP is installed and model is supported."""
    try:
        import shap  # type: ignore
    except Exception:
        return None

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    try:
        explainer = shap.Explainer(model, transformed)
        sv = explainer(transformed[row_idx : row_idx + 1])
        values = sv.values[0]
    except Exception:
        return None

    return pd.DataFrame({"feature": feature_names, "shap_value": values}).sort_values("shap_value", key=np.abs, ascending=False)



def local_reason_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback local explanations without SHAP."""
    featured = build_features(df)
    reasons = []
    for _, row in featured.iterrows():
        sample_reasons = []
        if row["amount_vs_avg_ratio"] > 3:
            sample_reasons.append("Amount is significantly above customer's 7-day baseline")
        if row["transaction_count_24h"] >= 10:
            sample_reasons.append("Unusually high transaction velocity in 24h")
        if row["geo_distance_from_home"] >= 300:
            sample_reasons.append("Transaction location far from home profile")
        if row["failed_login_count_24h"] >= 3:
            sample_reasons.append("Multiple recent failed login attempts")
        if row["beneficiary_age_days"] <= 7:
            sample_reasons.append("Very new beneficiary indicates possible mule account")
        reasons.append("; ".join(sample_reasons) or "No major anomaly signal")

    result = df.copy()
    result["reason_codes"] = reasons
    return result
