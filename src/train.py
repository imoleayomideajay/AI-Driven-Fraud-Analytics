"""Training pipeline for fraud detection models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.data_simulation import SimulationConfig, generate_synthetic_transactions, save_dataset
from src.evaluate import (
    capture_rate_at_top_n,
    compute_metrics,
    summarize_cv_results,
    tune_threshold_for_f1,
    tune_threshold_with_precision_floor,
)
from src.features import add_derived_features
from src.preprocess import build_preprocessor, split_features_target
from src.utils import MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED, ensure_directories, save_json, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    champion_name: str
    champion_threshold: float
    metrics: Dict[str, Dict[str, float]]
    cv_summary: Dict[str, Dict[str, float]]


def _get_boosting_model() -> Any:
    try:
        from xgboost import XGBClassifier

        logger.info("Using XGBoost model.")
        return XGBClassifier(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            scale_pos_weight=8,
        )
    except Exception:
        logger.warning("XGBoost unavailable; falling back to GradientBoostingClassifier.")
        return GradientBoostingClassifier(random_state=RANDOM_SEED)


def _model_candidates() -> Dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None),
        "random_forest": RandomForestClassifier(
            n_estimators=280,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "xgb_or_gb": _get_boosting_model(),
    }


def train_and_select_champion(df: pd.DataFrame) -> Tuple[Pipeline, TrainingArtifacts, pd.DataFrame]:
    """Train candidate models and return champion pipeline."""
    engineered = add_derived_features(df)
    X, y = split_features_target(engineered)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor()
    metrics_store: Dict[str, Dict[str, float]] = {}
    cv_store: Dict[str, Dict[str, float]] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, model in _model_candidates().items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        prob = pipeline.predict_proba(X_test)[:, 1]
        threshold, _ = tune_threshold_for_f1(y_test.values, prob)
        pred = (prob >= threshold).astype(int)

        model_metrics = compute_metrics(y_test.values, pred, prob)
        model_metrics["threshold"] = threshold
        model_metrics["capture_rate_top_5pct"] = capture_rate_at_top_n(y_test.values, prob, 0.05)
        metrics_store[name] = model_metrics

        cv = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED),
            scoring=["precision", "recall", "f1", "roc_auc", "average_precision"],
            n_jobs=-1,
        )
        cv_summary_df = summarize_cv_results({k: v for k, v in cv.items() if k.startswith("test_")})
        cv_store[name] = {
            row["metric"]: row["mean"] for _, row in cv_summary_df.iterrows()
        }

        fitted_pipelines[name] = pipeline
        logger.info("Model=%s metrics=%s", name, model_metrics)

    iso_preprocessor = build_preprocessor()
    X_train_prep = iso_preprocessor.fit_transform(X_train)
    X_test_prep = iso_preprocessor.transform(X_test)
    iso = IsolationForest(
        contamination=0.03,
        random_state=RANDOM_SEED,
        n_estimators=220,
        n_jobs=-1,
    )
    iso.fit(X_train_prep)
    iso_scores = -iso.decision_function(X_test_prep)
    iso_prob = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
    iso_threshold, _ = tune_threshold_for_f1(y_test.values, iso_prob)
    iso_pred = (iso_prob >= iso_threshold).astype(int)
    iso_metrics = compute_metrics(y_test.values, iso_pred, iso_prob)
    iso_metrics["threshold"] = iso_threshold
    iso_metrics["capture_rate_top_5pct"] = capture_rate_at_top_n(y_test.values, iso_prob, 0.05)
    metrics_store["isolation_forest"] = iso_metrics

    # Keep Isolation Forest as a benchmark only. It does not provide a calibrated
    # `predict_proba` interface and therefore cannot be used as the production
    # champion pipeline consumed by downstream scoring code.
    supervised_candidates = list(fitted_pipelines.keys())
    champion_name = max(
        supervised_candidates,
        key=lambda m: (metrics_store[m]["pr_auc"], metrics_store[m]["recall"]),
    )
    champion_threshold = metrics_store[champion_name]["threshold"]
    champion_pipeline = fitted_pipelines[champion_name]

    test_scored = X_test.copy()
    test_scored["label_fraud"] = y_test.values
    test_scored["fraud_probability"] = champion_pipeline.predict_proba(X_test)[:, 1]
    test_scored["fraud_prediction"] = (test_scored["fraud_probability"] >= champion_threshold).astype(int)

    artifacts = TrainingArtifacts(
        champion_name=champion_name,
        champion_threshold=champion_threshold,
        metrics=metrics_store,
        cv_summary=cv_store,
    )
    return champion_pipeline, artifacts, test_scored


def persist_training_outputs(
    champion_pipeline: Pipeline,
    artifacts: TrainingArtifacts,
    scored_test: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    """Persist model, metadata, and scored holdout output."""
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(champion_pipeline, models_dir / "champion_model.joblib")
    save_json(
        {
            "champion_model": artifacts.champion_name,
            "champion_threshold": artifacts.champion_threshold,
            "metrics": artifacts.metrics,
            "cv_summary": artifacts.cv_summary,
        },
        models_dir / "training_metadata.json",
    )
    scored_test.to_csv(outputs_dir / "test_scored.csv", index=False)


def run_training() -> None:
    """Generate data, train models, and save artifacts."""
    setup_logging()
    ensure_directories()
    df = generate_synthetic_transactions(SimulationConfig())
    save_dataset(df)
    champion_pipeline, artifacts, scored = train_and_select_champion(df)
    persist_training_outputs(champion_pipeline, artifacts, scored)
    logger.info("Champion model: %s", artifacts.champion_name)


if __name__ == "__main__":
    run_training()
