"""Convenience loaders for the manuscript datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


@dataclass(frozen=True)
class PaperDatasetSpec:
    """Pinned dataset metadata used by the manuscript experiments."""

    key: str
    name: str
    openml_id: int | None
    openml_name: str | None
    openml_version: int | None
    target: str
    n_rows: int
    minority_value: object | None
    classification_columns: tuple[str, ...]
    regression_columns: tuple[str, ...]
    ignore_columns: tuple[str, ...] = ()


ADULT_CLASSIFICATION = (
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
)
ADULT_REGRESSION = ("age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week")

BANK_CLASSIFICATION = ("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome")
BANK_REGRESSION = ("age", "balance", "day", "duration", "campaign", "pdays", "previous")

DEFAULT_CREDIT_CLASSIFICATION = ("SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
DEFAULT_CREDIT_REGRESSION = (
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


PAPER_DATASETS: dict[str, PaperDatasetSpec] = {
    "adult": PaperDatasetSpec(
        key="adult",
        name="Adult",
        openml_id=1590,
        openml_name="adult",
        openml_version=2,
        target="class",
        n_rows=48842,
        minority_value=">50K",
        classification_columns=ADULT_CLASSIFICATION,
        regression_columns=ADULT_REGRESSION,
    ),
    "bank_marketing": PaperDatasetSpec(
        key="bank_marketing",
        name="Bank Marketing",
        openml_id=44234,
        openml_name="Bank_marketing_data_set_UCI",
        openml_version=None,
        target="y",
        n_rows=45211,
        minority_value="yes",
        classification_columns=BANK_CLASSIFICATION,
        regression_columns=BANK_REGRESSION,
    ),
    "default_credit": PaperDatasetSpec(
        key="default_credit",
        name="Default of Credit Card Clients",
        openml_id=42477,
        openml_name="default-of-credit-card-clients",
        openml_version=None,
        target="y",
        n_rows=30000,
        minority_value=1,
        classification_columns=DEFAULT_CREDIT_CLASSIFICATION,
        regression_columns=DEFAULT_CREDIT_REGRESSION,
        ignore_columns=("ID",),
    ),
}

_ALIASES = {
    "adult_mixed": "adult",
    "bank": "bank_marketing",
    "bank-marketing": "bank_marketing",
    "credit": "default_credit",
    "default": "default_credit",
    "default-of-credit-card-clients": "default_credit",
}


def paper_dataset_names() -> list[str]:
    """Return the public keys accepted by :func:`load_paper_dataset`."""

    return list(PAPER_DATASETS)


def paper_dataset_registry() -> pd.DataFrame:
    """Return metadata for the three pinned manuscript datasets."""

    rows = []
    for spec in PAPER_DATASETS.values():
        rows.append(
            {
                "key": spec.key,
                "name": spec.name,
                "openml_id": spec.openml_id,
                "target": spec.target,
                "n_rows": spec.n_rows,
                "regression_columns": len(spec.regression_columns),
                "classification_columns": len(spec.classification_columns) + 1,
                "ignore_columns": len(spec.ignore_columns),
            }
        )
    return pd.DataFrame(rows)


def load_paper_dataset(
    key: str,
    *,
    n_rows: int | None = None,
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load one manuscript dataset and return ``(df, columns)`` for MIMIC.

    The returned dataframe has a standardized binary ``label`` column with values
    ``"minority"`` and ``"majority"``. The returned ``columns`` dictionary uses
    MIMIC's public role names: ``ignore``, ``regression``, and
    ``classification``.
    """

    resolved = _resolve_key(key)
    spec = PAPER_DATASETS[resolved]
    frame = _load_openml_frame(spec)
    frame = _standardize_frame(frame, spec)
    if n_rows is not None:
        n_rows = int(n_rows)
        if n_rows <= 0:
            raise ValueError("n_rows must be positive when provided")
        if n_rows < len(frame):
            frame = _stratified_subsample(frame, n_rows=n_rows, random_state=random_state)
    frame = frame.reset_index(drop=True)
    return frame, paper_dataset_columns(frame, resolved)


def paper_dataset_columns(frame: pd.DataFrame, key: str) -> dict[str, list[str]]:
    """Return MIMIC column roles for a loaded manuscript dataset."""

    spec = PAPER_DATASETS[_resolve_key(key)]
    ignore = [column for column in spec.ignore_columns if column in frame.columns]
    regression = [column for column in spec.regression_columns if column in frame.columns and column != "label"]
    classification = [
        column for column in spec.classification_columns if column in frame.columns and column not in {"label", *ignore}
    ]
    if "label" in frame.columns:
        classification.append("label")

    assigned = set(ignore + regression + classification)
    extra = [column for column in frame.columns if column not in assigned]
    for column in extra:
        if pd.api.types.is_numeric_dtype(frame[column]) and frame[column].nunique(dropna=True) > 10:
            regression.append(column)
        else:
            classification.append(column)
    return {"ignore": ignore, "regression": regression, "classification": classification}


def _resolve_key(key: str) -> str:
    normalized = str(key).strip().lower().replace(" ", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in PAPER_DATASETS:
        choices = ", ".join(sorted(PAPER_DATASETS))
        raise ValueError(f"Unknown paper dataset {key!r}; expected one of {choices}")
    return normalized


def _load_openml_frame(spec: PaperDatasetSpec) -> pd.DataFrame:
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


def _standardize_frame(frame: pd.DataFrame, spec: PaperDatasetSpec) -> pd.DataFrame:
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
    lower = {str(column).lower(): column for column in frame.columns}
    if target.lower() in lower:
        return lower[target.lower()]
    candidates = [column for column in frame.columns if str(column).lower() in {"class", "target", "y"}]
    if candidates:
        return candidates[-1]
    raise ValueError(f"Target column {target!r} not found")


def _stratified_subsample(frame: pd.DataFrame, *, n_rows: int, random_state: int) -> pd.DataFrame:
    if "label" not in frame.columns:
        return frame.sample(n=n_rows, random_state=random_state).reset_index(drop=True)
    fractions = frame["label"].value_counts(normalize=True)
    pieces = []
    remaining = n_rows
    labels = list(fractions.index)
    for index, label in enumerate(labels):
        group = frame.loc[frame["label"].eq(label)]
        if index == len(labels) - 1:
            size = min(len(group), remaining)
        else:
            size = min(len(group), max(1, int(round(n_rows * float(fractions[label])))))
            remaining -= size
        pieces.append(group.sample(n=size, random_state=random_state))
    return pd.concat(pieces).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
