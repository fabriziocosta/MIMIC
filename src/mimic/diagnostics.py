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
    log1p_features: list[str] | None = None,
    original_label: str = "original",
    generated_label: str = "generated",
    max_rows_per_source: int | None = 300,
    max_hist_bins: int = 20,
    random_state=None,
):
    """Plot triangular pairwise feature statistics for original and generated rows."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    original_plot = original[features].copy()
    generated_plot = generated[features].copy()
    display_features = list(features)
    if log1p_features is not None:
        log1p_set = set(log1p_features)
        for feature in features:
            if feature in log1p_set:
                display_name = f"log1p({feature})"
                original_plot[display_name] = np.log1p(pd.to_numeric(original_plot[feature], errors="coerce"))
                generated_plot[display_name] = np.log1p(pd.to_numeric(generated_plot[feature], errors="coerce"))
                original_plot = original_plot.drop(columns=[feature])
                generated_plot = generated_plot.drop(columns=[feature])
                display_features[display_features.index(feature)] = display_name
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

    palette = dict(zip([original_label, generated_label], sns.color_palette(n_colors=2)))
    grid = sns.PairGrid(
        plot_data,
        vars=display_features,
        hue="source",
        corner=True,
        height=2.2,
        palette=palette,
        diag_sharey=False,
    )
    grid.map_lower(plt.scatter, alpha=0.45, s=18, edgecolor="none")
    _map_log1p_filled_histograms(
        grid,
        plot_data,
        display_features,
        [original_label, generated_label],
        palette,
        max_bins=max_hist_bins,
    )
    grid.add_legend()
    return grid


def categorical_feature_report(
    original: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    features: list[str],
    original_label: str = "original",
    generated_label: str = "generated",
    top_n: int = 12,
    missing_label: str = "__missing__",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare categorical feature distributions for original and generated rows."""
    rows = []
    summaries = []
    for feature in features:
        original_values = original[feature].astype("object").where(original[feature].notna(), missing_label)
        generated_values = generated[feature].astype("object").where(generated[feature].notna(), missing_label)
        categories = pd.Index(pd.unique(original_values)).union(pd.Index(pd.unique(generated_values)))
        original_props = original_values.value_counts(normalize=True).reindex(categories, fill_value=0.0)
        generated_props = generated_values.value_counts(normalize=True).reindex(categories, fill_value=0.0)
        diffs = (generated_props - original_props).abs()
        tvd = 0.5 * float(diffs.sum())
        summaries.append({"feature": feature, "total_variation_distance": tvd, "n_categories": len(categories)})
        top_categories = diffs.sort_values(ascending=False).head(top_n).index
        for category in top_categories:
            rows.append(
                {
                    "feature": feature,
                    "category": category,
                    f"{original_label}_proportion": float(original_props.loc[category]),
                    f"{generated_label}_proportion": float(generated_props.loc[category]),
                    "absolute_difference": float(diffs.loc[category]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def categorical_feature_plot(
    original: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    features: list[str],
    original_label: str = "original",
    generated_label: str = "generated",
    top_n: int = 12,
):
    """Plot top categorical proportion differences for original and generated rows."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    report, summary = categorical_feature_report(
        original,
        generated,
        features=features,
        original_label=original_label,
        generated_label=generated_label,
        top_n=top_n,
    )
    if report.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_axis_off()
        return fig, np.asarray([ax]), report, summary

    value_vars = [f"{original_label}_proportion", f"{generated_label}_proportion"]
    plot_data = report.melt(
        id_vars=["feature", "category"],
        value_vars=value_vars,
        var_name="source",
        value_name="proportion",
    )
    plot_data["source"] = plot_data["source"].str.replace("_proportion", "", regex=False)
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(9, max(3, 2.8 * n_features)), squeeze=False)
    axes = axes.ravel()
    for ax, feature in zip(axes, features):
        part = plot_data[plot_data["feature"] == feature]
        sns.barplot(part, x="proportion", y="category", hue="source", ax=ax)
        tvd = summary.loc[summary["feature"] == feature, "total_variation_distance"].iat[0]
        ax.set_title(f"{feature} category proportions (TVD={tvd:.3f})")
        ax.set_xlabel("proportion")
        ax.set_ylabel("")
    fig.tight_layout()
    return fig, axes, report, summary


def _map_log1p_filled_histograms(grid, plot_data, features, labels, palette, max_bins: int = 20):
    for i, feature in enumerate(features):
        base_ax = grid.axes[i, i]
        if base_ax is None:
            continue
        ax = base_ax.twinx()
        base_ax.set_yticks([])
        values = pd.to_numeric(plot_data[feature], errors="coerce").dropna()
        if values.empty:
            continue
        auto_bins = np.histogram_bin_edges(values.to_numpy(), bins="auto")
        if max_bins is not None and len(auto_bins) - 1 > max_bins:
            bins = np.linspace(auto_bins[0], auto_bins[-1], max_bins + 1)
        else:
            bins = auto_bins
        if len(bins) < 2:
            center = float(values.iloc[0])
            bins = np.array([center - 0.5, center + 0.5])
        for label in labels:
            series = pd.to_numeric(plot_data.loc[plot_data["source"] == label, feature], errors="coerce").dropna()
            counts, edges = np.histogram(series.to_numpy(), bins=bins)
            ax.bar(
                edges[:-1],
                np.log1p(counts),
                width=np.diff(edges),
                align="edge",
                alpha=0.45,
                color=palette[label],
                edgecolor=palette[label],
                linewidth=0.6,
                label=label,
            )
        ax.set_ylabel("log1p(count)")


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
