from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
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


def classical_mds_2d(X, *, center=None, random_state=None) -> pd.DataFrame:
    """Project a numeric matrix with classical metric MDS."""
    X = np.asarray(X, dtype=float)
    if len(X) == 0:
        return pd.DataFrame(columns=["mds1", "mds2"])

    if center is None:
        c = np.nanmean(X, axis=0)
    elif isinstance(center, str) and center == "random":
        rng = np.random.default_rng(random_state)
        c = X[int(rng.integers(0, len(X)))]
    elif isinstance(center, (int, np.integer)):
        c = X[int(center)]
    else:
        c = np.asarray(center, dtype=float)
        if c.shape[0] != X.shape[1]:
            raise ValueError("Explicit center dimensionality must match X")

    D2 = pairwise_distances(X - c, metric="euclidean", squared=True)
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:2]
    vals = np.maximum(vals[order], 0)
    coords = vecs[:, order] * np.sqrt(vals)
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    return pd.DataFrame(coords, columns=["mds1", "mds2"])


def pairwise_feature_plot(
    original: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    features: list[str],
    original_label: str = "original",
    generated_label: str = "generated",
    max_rows_per_source: int | None = 300,
    random_state=None,
):
    """Plot triangular pairwise feature statistics for original and generated rows."""
    import seaborn as sns

    original_plot = original[features].copy()
    generated_plot = generated[features].copy()
    if max_rows_per_source is not None:
        original_plot = original_plot.sample(
            n=min(max_rows_per_source, len(original_plot)),
            random_state=random_state,
        )
        generated_plot = generated_plot.sample(
            n=min(max_rows_per_source, len(generated_plot)),
            random_state=random_state,
        )

    original_plot["source"] = original_label
    generated_plot["source"] = generated_label
    plot_data = pd.concat([original_plot, generated_plot], ignore_index=True)

    return sns.pairplot(
        plot_data,
        vars=features,
        hue="source",
        corner=True,
        diag_kind="hist",
        height=2.2,
        plot_kws={"alpha": 0.45, "s": 18, "edgecolor": "none"},
        diag_kws={
            "alpha": 0.75,
            "common_norm": False,
            "element": "step",
            "fill": False,
            "log_scale": (False, True),
        },
    )


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
