"""Train-only preprocessing for streamlined experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .datasets import feature_roles


@dataclass
class PreparedData:
    preprocessor: ColumnTransformer
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    numeric_columns: list[str]
    categorical_columns: list[str]
    feature_names: list[str]


def fit_preprocess_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    dataset_key: str,
    target_column: str = "label",
) -> PreparedData:
    roles = feature_roles(train, dataset_key)
    numeric = roles["numeric"]
    categorical = roles["categorical"]
    preprocess = build_preprocessor(numeric, categorical)
    X_train_raw = train[numeric + categorical]
    X_test_raw = test[numeric + categorical]
    X_train = _dense(preprocess.fit_transform(X_train_raw))
    X_test = _dense(preprocess.transform(X_test_raw))
    y_train = train[target_column].eq("minority").astype(int).to_numpy()
    y_test = test[target_column].eq("minority").astype(int).to_numpy()
    return PreparedData(
        preprocessor=preprocess,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numeric_columns=numeric,
        categorical_columns=categorical,
        feature_names=_feature_names(preprocess, numeric, categorical),
    )


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def transform_frame(preprocessor: ColumnTransformer, frame: pd.DataFrame, *, numeric_columns: list[str], categorical_columns: list[str]) -> np.ndarray:
    return _dense(preprocessor.transform(frame[numeric_columns + categorical_columns]))


def _dense(values) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=float)


def _feature_names(preprocessor: ColumnTransformer, numeric: list[str], categorical: list[str]) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return numeric + categorical
