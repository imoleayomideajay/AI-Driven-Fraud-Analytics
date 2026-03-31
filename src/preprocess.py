"""Preprocessing pipeline definitions."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import build_features

TARGET_COLUMN = "label_fraud"
ID_COLUMNS = ["transaction_id", "customer_id", "timestamp"]


NUMERIC_FEATURES = [
    "ip_risk_score",
    "geo_distance_from_home",
    "amount",
    "account_age_days",
    "avg_transaction_amount_7d",
    "transaction_count_24h",
    "failed_login_count_24h",
    "beneficiary_age_days",
    "velocity_score",
    "prior_fraud_flag",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_night",
    "hour_sin",
    "hour_cos",
    "amount_vs_avg_ratio",
    "geo_velocity_interaction",
    "login_velocity_interaction",
    "risk_composite",
]

CATEGORICAL_FEATURES = ["transaction_type", "channel", "merchant_category", "device_type"]



def prepare_modeling_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build model-ready feature matrix and target."""
    featured = build_features(df)
    X = featured.drop(columns=[TARGET_COLUMN] + ID_COLUMNS)
    y = featured[TARGET_COLUMN]
    return X, y



def get_preprocessor() -> ColumnTransformer:
    """Create preprocessing transformer for numerical + categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
