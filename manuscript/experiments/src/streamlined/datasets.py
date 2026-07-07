"""Pinned dataset registry and loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    openml_id: int | None
    openml_name: str | None
    openml_version: int | None
    target: str
    n_rows: int
    minority_value: object | None = None
    categorical_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    ignore_columns: tuple[str, ...] = ()
    loader: Callable[["DatasetSpec", int | None, int], pd.DataFrame] | None = None


ADULT_CATEGORICAL = (
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
)
ADULT_NUMERIC = ("age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week")

BANK_CATEGORICAL = ("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome")
BANK_NUMERIC = ("age", "balance", "day", "duration", "campaign", "pdays", "previous")

DEFAULT_CATEGORICAL = ("SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
DEFAULT_NUMERIC = (
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
)


DATASETS = {
    "adult": DatasetSpec(
        key="adult",
        name="Adult",
        openml_id=1590,
        openml_name="adult",
        openml_version=2,
        target="class",
        n_rows=48842,
        minority_value=">50K",
        categorical_columns=ADULT_CATEGORICAL,
        numeric_columns=ADULT_NUMERIC,
    ),
    "bank_marketing": DatasetSpec(
        key="bank_marketing",
        name="Bank Marketing",
        openml_id=44234,
        openml_name="Bank_marketing_data_set_UCI",
        openml_version=None,
        target="y",
        n_rows=45211,
        minority_value="yes",
        categorical_columns=BANK_CATEGORICAL,
        numeric_columns=BANK_NUMERIC,
    ),
    "default_credit": DatasetSpec(
        key="default_credit",
        name="Default of Credit Card Clients",
        openml_id=42477,
        openml_name="default-of-credit-card-clients",
        openml_version=None,
        target="y",
        n_rows=30000,
        minority_value=1,
        categorical_columns=DEFAULT_CATEGORICAL,
        numeric_columns=DEFAULT_NUMERIC,
        ignore_columns=("ID",),
    ),
}


def dataset_registry() -> pd.DataFrame:
    rows = []
    for spec in DATASETS.values():
        rows.append(
            {
                "key": spec.key,
                "name": spec.name,
                "openml_id": spec.openml_id,
                "target": spec.target,
                "n_rows": spec.n_rows,
                "numeric_features": len(spec.numeric_columns),
                "categorical_features": len(spec.categorical_columns),
                "status": "ready",
            }
        )
    return pd.DataFrame(rows)


def load_dataset(key: str, *, n_rows: int | None = None, random_state: int = 0) -> pd.DataFrame:
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset key: {key!r}")
    spec = DATASETS[key]
    if spec.loader is not None:
        frame = spec.loader(spec, n_rows, random_state)
    else:
        frame = _load_openml(spec)
    frame = _standardize_frame(frame, spec)
    if n_rows is not None and n_rows < len(frame):
        frame, _unused = _stratified_subsample(frame, n_rows=n_rows, random_state=random_state)
    return frame.reset_index(drop=True)


def feature_roles(frame: pd.DataFrame, key: str) -> dict[str, list[str]]:
    spec = DATASETS[key]
    ignore = [c for c in spec.ignore_columns if c in frame.columns]
    categorical = [c for c in spec.categorical_columns if c in frame.columns and c != "label"]
    numeric = [c for c in spec.numeric_columns if c in frame.columns and c != "label"]
    used = set(ignore + categorical + numeric + ["label"])
    extra = [c for c in frame.columns if c not in used]
    for column in extra:
        if pd.api.types.is_numeric_dtype(frame[column]) and frame[column].nunique(dropna=True) > 10:
            numeric.append(column)
        else:
            categorical.append(column)
    return {"ignore": ignore, "numeric": numeric, "categorical": categorical}


def dataset_metadata(frame: pd.DataFrame, key: str, *, preprocessed_dim: int | None = None, baseline_roc_auc: float | None = None) -> dict:
    roles = feature_roles(frame, key)
    counts = frame["label"].value_counts()
    majority = int(counts.get("majority", 0))
    minority = int(counts.get("minority", 0))
    return {
        "dataset_key": key,
        "n_samples": int(len(frame)),
        "n_features": int(len(frame.columns) - 1),
        "n_numeric_features": len(roles["numeric"]),
        "n_categorical_features": len(roles["categorical"]),
        "original_imbalance_ratio": float(majority / minority) if minority else np.nan,
        "minority_size": minority,
        "preprocessed_dim": preprocessed_dim,
        "missingness_rate": float(frame.drop(columns=["label"]).isna().to_numpy().mean()),
        "baseline_roc_auc": baseline_roc_auc,
    }


def _load_openml(spec: DatasetSpec) -> pd.DataFrame:
    try:
        if spec.openml_id is not None:
            dataset = fetch_openml(data_id=spec.openml_id, as_frame=True)
        else:
            dataset = fetch_openml(name=spec.openml_name, version=spec.openml_version, as_frame=True)
    except Exception:
        if spec.openml_name is None:
            raise
        dataset = fetch_openml(name=spec.openml_name, version=spec.openml_version, as_frame=True)
    if getattr(dataset, "frame", None) is not None:
        return dataset.frame.copy()
    frame = pd.DataFrame(dataset.data).copy()
    frame[spec.target] = dataset.target
    return frame


def _standardize_frame(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    frame = frame.copy().replace("?", np.nan)
    target = _resolve_target(frame, spec.target)
    labels = frame[target]
    features = frame.drop(columns=[target])
    if spec.minority_value is None:
        counts = labels.value_counts(dropna=True)
        if len(counts) != 2:
            raise ValueError(f"Expected binary target for {spec.key}, found {len(counts)} classes")
        minority_value = counts.idxmin()
    else:
        minority_value = spec.minority_value
    normalized = labels.astype(str).eq(str(minority_value))
    features["label"] = np.where(normalized, "minority", "majority")
    return features


def _resolve_target(frame: pd.DataFrame, target: str) -> str:
    if target in frame.columns:
        return target
    lower = {str(c).lower(): c for c in frame.columns}
    if target.lower() in lower:
        return lower[target.lower()]
    candidates = [c for c in frame.columns if str(c).lower() in {"class", "target", "y"}]
    if candidates:
        return candidates[-1]
    raise ValueError(f"Target column {target!r} not found")


def _stratified_subsample(frame: pd.DataFrame, *, n_rows: int, random_state: int):
    if n_rows >= len(frame):
        return frame.reset_index(drop=True), None
    fractions = frame["label"].value_counts(normalize=True)
    pieces = []
    remaining = n_rows
    labels = list(fractions.index)
    for i, label in enumerate(labels):
        group = frame.loc[frame["label"].eq(label)]
        if i == len(labels) - 1:
            size = min(len(group), remaining)
        else:
            size = min(len(group), max(1, int(round(n_rows * float(fractions[label])))))
            remaining -= size
        pieces.append(group.sample(n=size, random_state=random_state))
    sampled = pd.concat(pieces).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled, None
