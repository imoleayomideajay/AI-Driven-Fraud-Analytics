"""Model training pipeline for fraud analytics."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.data_simulation import create_and_save_dataset
from src.evaluate import compute_metrics, metrics_table, tune_threshold_for_f1
from src.preprocess import build_preprocessor, prepare_data
from src.utils import MODELS_DIR, OUTPUTS_DIR, ensure_directories, save_artifact, save_json, setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainedBundle:
    champion_name: str
    champion_threshold: float
    model_scores: Dict[str, Dict[str, float]]



def _get_models() -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=240,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }



def train_and_evaluate(df: pd.DataFrame) -> TrainedBundle:
    """Train all supervised models + isolation forest benchmark and persist artifacts."""
    prepared = prepare_data(df)
    preprocessor, _, _ = build_preprocessor(prepared.X_train)

    results: Dict[str, Dict[str, float]] = {}
    artifacts: Dict[str, Dict[str, object]] = {}

    for name, estimator in _get_models().items():
        pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        cv_pr_auc = cross_val_score(pipe, prepared.X_train, prepared.y_train, cv=cv, scoring="average_precision")
        pipe.fit(prepared.X_train, prepared.y_train)

        proba = pipe.predict_proba(prepared.X_test)[:, 1]
        threshold, _ = tune_threshold_for_f1(prepared.y_test.values, proba)
        metrics = compute_metrics(prepared.y_test.values, proba, threshold=threshold)
        metrics["cv_pr_auc_mean"] = float(np.mean(cv_pr_auc))
        metrics["threshold"] = threshold

        results[name] = metrics
        artifacts[name] = {"pipeline": pipe, "threshold": threshold}
        LOGGER.info("Model %s metrics: %s", name, metrics)

    # Isolation forest benchmark
    iso_pipe = Pipeline(
        [
            ("preprocess", clone(preprocessor)),
            (
                "model",
                IsolationForest(
                    n_estimators=250,
                    contamination=max(0.005, float(prepared.y_train.mean())),
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    iso_pipe.fit(prepared.X_train)
    iso_scores = -iso_pipe.decision_function(prepared.X_test)
    iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
    iso_t, _ = tune_threshold_for_f1(prepared.y_test.values, iso_norm)
    iso_metrics = compute_metrics(prepared.y_test.values, iso_norm, threshold=iso_t)
    iso_metrics["cv_pr_auc_mean"] = float("nan")
    iso_metrics["threshold"] = iso_t
    results["isolation_forest"] = iso_metrics
    artifacts["isolation_forest"] = {"pipeline": iso_pipe, "threshold": iso_t}

    scores_df = metrics_table(results)
    champion_name = scores_df.index[0]
    champion_threshold = float(results[champion_name]["threshold"])

    save_artifact(artifacts[champion_name]["pipeline"], MODELS_DIR / "champion_pipeline.joblib")
    save_artifact(
        {
            "threshold": champion_threshold,
            "model_name": champion_name,
            "feature_columns": prepared.feature_columns,
        },
        MODELS_DIR / "metadata.joblib",
    )

    for model_name, payload in artifacts.items():
        save_artifact(payload["pipeline"], MODELS_DIR / f"{model_name}.joblib")

    scores_df.to_csv(OUTPUTS_DIR / "model_metrics.csv")
    save_json(results, OUTPUTS_DIR / "model_metrics.json")
    prepared.full_df.to_csv(OUTPUTS_DIR / "feature_enriched_dataset.csv", index=False)

    return TrainedBundle(champion_name=champion_name, champion_threshold=champion_threshold, model_scores=results)


if __name__ == "__main__":
    setup_logging()
    ensure_directories()
    frame, _ = create_and_save_dataset(60_000)
    bundle = train_and_evaluate(frame)
    LOGGER.info("Champion model: %s @ threshold %.3f", bundle.champion_name, bundle.champion_threshold)
