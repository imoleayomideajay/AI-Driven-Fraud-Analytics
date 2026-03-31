"""Preprocessing pipeline construction and dataset split helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import engineer_features

TARGET = "label_fraud"
ID_COLUMNS = ["transaction_id", "customer_id", "timestamp"]


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    full_df: pd.DataFrame
    feature_columns: List[str]



def build_preprocessor(X: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    """Build sklearn preprocessing pipeline with numeric scaling + OHE."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols



def prepare_data(df: pd.DataFrame) -> PreparedData:
    """Engineer features then split chronologically to mimic production deployment."""
    fe_df = engineer_features(df)
    fe_df = fe_df.sort_values("timestamp").reset_index(drop=True)

    features = [c for c in fe_df.columns if c not in ID_COLUMNS + [TARGET]]
    X = fe_df[features]
    y = fe_df[TARGET].astype(int)

    split_idx = int(len(fe_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        full_df=fe_df,
        feature_columns=features,
    )
