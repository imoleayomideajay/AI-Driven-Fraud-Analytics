"""Preprocessing pipeline for fraud detection models."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import feature_columns


TARGET_COLUMN = "label_fraud"
DROP_COLUMNS = ["transaction_id", "customer_id", "timestamp"]


def build_preprocessor() -> ColumnTransformer:
    """Build column transformer for numerical and categorical variables."""
    numeric_features, categorical_features = feature_columns()

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", numeric_pipe, numeric_features), ("cat", categorical_pipe, categorical_features)]
    )


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into model features and target."""
    model_df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).copy()
    y = model_df[TARGET_COLUMN].astype(int)
    X = model_df.drop(columns=[TARGET_COLUMN])
    return X, y
