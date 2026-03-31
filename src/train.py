"""Training script for fraud detection models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.evaluate import eval_result_to_dict, evaluate_predictions, select_threshold_for_f1
from src.preprocess import get_preprocessor, prepare_modeling_data
from src.utils import ensure_directories, setup_logging

LOGGER = logging.getLogger(__name__)
RANDOM_SEED = 42



def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run src/data_simulation.py first.")
    return pd.read_csv(path, parse_dates=["timestamp"])



def build_models() -> Dict[str, object]:
    """Create model candidates."""
    return {
        "logistic_regression": LogisticRegression(max_iter=600, class_weight="balanced", random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=240,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }



def fit_isolation_forest(X_train: pd.DataFrame) -> IsolationForest:
    """Fit anomaly baseline using only non-fraud labels is often ideal; here fit unsupervised on train mix."""
    model = IsolationForest(
        n_estimators=250,
        contamination=0.03,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train.select_dtypes(include=["number"]))
    return model



def train_and_evaluate(data_path: Path, models_dir: Path, outputs_dir: Path) -> Tuple[pd.DataFrame, dict]:
    """Train candidate models, compare metrics, persist champion."""
    df = load_data(data_path)
    X, y = prepare_modeling_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model_results = []
    champion_name = ""
    champion_score = -1.0
    champion_bundle = {}

    for name, model in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", get_preprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]

        best_threshold = select_threshold_for_f1(y_test.to_numpy(), probs)
        result = evaluate_predictions(name, y_test.to_numpy(), probs, best_threshold)
        metrics = eval_result_to_dict(result)

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
        cv_pr_auc = cross_val_score(pipeline, X_train, y_train, scoring="average_precision", cv=cv, n_jobs=-1)
        metrics["cv_pr_auc_mean"] = float(np.mean(cv_pr_auc))
        metrics["cv_pr_auc_std"] = float(np.std(cv_pr_auc))
        model_results.append(metrics)

        if metrics["pr_auc"] > champion_score:
            champion_name = name
            champion_score = metrics["pr_auc"]
            champion_bundle = {
                "pipeline": pipeline,
                "threshold": best_threshold,
                "metrics": metrics,
            }

    # Isolation forest benchmark
    iso_model = fit_isolation_forest(X_train)
    iso_scores = -iso_model.score_samples(X_test.select_dtypes(include=["number"]))
    iso_scores = (iso_scores - iso_scores.min()) / max(iso_scores.max() - iso_scores.min(), 1e-8)
    iso_threshold = select_threshold_for_f1(y_test.to_numpy(), iso_scores)
    iso_result = eval_result_to_dict(evaluate_predictions("isolation_forest", y_test.to_numpy(), iso_scores, iso_threshold))
    model_results.append(iso_result)

    results_df = pd.DataFrame(model_results).sort_values("pr_auc", ascending=False)

    ensure_directories([models_dir, outputs_dir])
    results_df.to_csv(outputs_dir / "model_comparison.csv", index=False)

    joblib.dump(champion_bundle["pipeline"], models_dir / "champion_pipeline.joblib")
    joblib.dump(champion_bundle["threshold"], models_dir / "champion_threshold.joblib")
    joblib.dump(X_train.columns.tolist(), models_dir / "feature_columns.joblib")

    with open(models_dir / "train_summary.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "champion_model": champion_name,
                "champion_metrics": champion_bundle["metrics"],
                "rows_train": int(len(X_train)),
                "rows_test": int(len(X_test)),
                "fraud_rate_train": float(y_train.mean()),
                "fraud_rate_test": float(y_test.mean()),
            },
            fp,
            indent=2,
        )

    LOGGER.info("Champion model selected: %s", champion_name)
    return results_df, champion_bundle


if __name__ == "__main__":
    setup_logging()
    results, champion = train_and_evaluate(
        data_path=Path("data/synthetic_transactions.csv"),
        models_dir=Path("models"),
        outputs_dir=Path("outputs"),
    )
    print(results.head())
