"""Explainability helpers using SHAP when available with fallback option."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.features import engineer_features
from src.predict import load_model_assets

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None



def global_feature_importance(sample_df: pd.DataFrame, sample_size: int = 2000) -> pd.DataFrame:
    """Compute global feature importance (model-native or permutation)."""
    assets = load_model_assets()
    pipeline = assets["pipeline"]
    metadata = assets["metadata"]
    fe = engineer_features(sample_df).sample(min(sample_size, len(sample_df)), random_state=42)
    X = fe[metadata["feature_columns"]]

    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    Xt = preprocess.transform(X)

    if hasattr(model, "feature_importances_"):
        names = preprocess.get_feature_names_out()
        imp = model.feature_importances_
        return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)

    if hasattr(model, "coef_"):
        names = preprocess.get_feature_names_out()
        imp = np.abs(model.coef_).ravel()
        return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)

    y_hat = (pipeline.predict(X) > 0).astype(int)
    perm = permutation_importance(model, Xt, y_hat, n_repeats=5, random_state=42)
    return pd.DataFrame({"feature": preprocess.get_feature_names_out(), "importance": perm.importances_mean}).sort_values(
        "importance", ascending=False
    )



def explain_single_prediction(tx_df: pd.DataFrame) -> Dict[str, object]:
    """Return local explanation as SHAP values when possible."""
    assets = load_model_assets()
    pipeline = assets["pipeline"]
    metadata = assets["metadata"]

    fe = engineer_features(tx_df)
    X = fe[metadata["feature_columns"]]

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        score = float(pipeline.predict_proba(X)[:, 1][0])
    else:
        raw = -pipeline.decision_function(X)
        scaled = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        score = float(scaled[0])

    if shap is not None:
        preprocess = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        Xt = preprocess.transform(X)
        feature_names = preprocess.get_feature_names_out()
        try:
            explainer = shap.Explainer(model, Xt)
            sv = explainer(Xt)
            contrib = sv.values[0]
            top_idx = np.argsort(np.abs(contrib))[::-1][:8]
            drivers = [{"feature": feature_names[i], "impact": float(contrib[i])} for i in top_idx]
            return {"score": score, "method": "shap", "drivers": drivers}
        except Exception:
            pass

    global_imp = global_feature_importance(tx_df, sample_size=len(tx_df)).head(8)
    drivers = [{"feature": r["feature"], "impact": float(r["importance"])} for _, r in global_imp.iterrows()]
    return {"score": score, "method": "fallback_importance", "drivers": drivers}
