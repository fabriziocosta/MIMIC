"""Metric helpers for streamlined experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score


METRIC_COLUMNS = ("roc_auc", "pr_auc", "balanced_accuracy", "f1", "brier")


def evaluate_binary_classifier(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_pred is None:
        y_pred = (y_score >= 0.5).astype(int)
    return {
        "roc_auc": _safe_metric(roc_auc_score, y_true, y_score),
        "pr_auc": _safe_metric(average_precision_score, y_true, y_score),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "f1": _safe_metric(f1_score, y_true, y_pred),
        "brier": _safe_metric(brier_score_loss, y_true, y_score),
    }


def result_schema() -> list[str]:
    return [
        "dataset_key",
        "imbalance_ratio",
        "training_size",
        "seed",
        "method",
        "generated_count",
        "real_minority_count",
        "real_majority_count",
        *METRIC_COLUMNS,
    ]


def empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=result_schema())


def _safe_metric(fn, *args) -> float:
    try:
        return float(fn(*args))
    except ValueError:
        return float("nan")
