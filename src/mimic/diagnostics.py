from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class BinaryClassificationDiagnostics:
    metrics: pd.DataFrame
    roc_curve: pd.DataFrame
    pr_curve: pd.DataFrame
    confusion_counts: pd.DataFrame
    confusion_relative: pd.DataFrame
    positive_rate: float


def binary_classification_diagnostics(
    y_true,
    y_pred,
    confidence: pd.DataFrame,
    *,
    positive_label: str,
    negative_label: str,
) -> BinaryClassificationDiagnostics:
    """Build notebook-friendly binary classification diagnostics."""
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    labels = [negative_label, positive_label]

    scores = (
        confidence["probabilities"]
        .map(lambda probs: float(probs.get(positive_label, 0.0)) if isinstance(probs, dict) else np.nan)
        .to_numpy()
    )
    y_binary = (y_true == positive_label).astype(int).to_numpy()

    fpr, tpr, _ = roc_curve(y_binary, scores)
    precision, recall, _ = precision_recall_curve(y_binary, scores)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    counts = pd.DataFrame(cm, index=pd.Index(labels, name="actual"), columns=pd.Index(labels, name="predicted"))

    metrics = pd.DataFrame(
        {
            "metric": ["accuracy", "macro_f1", "roc_auc", "average_precision"],
            "value": [
                accuracy_score(y_true, y_pred),
                f1_score(y_true, y_pred, average="macro"),
                roc_auc_score(y_binary, scores),
                average_precision_score(y_binary, scores),
            ],
        }
    )

    return BinaryClassificationDiagnostics(
        metrics=metrics,
        roc_curve=pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        pr_curve=pd.DataFrame({"recall": recall, "precision": precision}),
        confusion_counts=counts,
        confusion_relative=counts / counts.to_numpy().sum(),
        positive_rate=float(y_binary.mean()),
    )
