"""Plotting helpers for streamlined experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curves(learning: pd.DataFrame, *, dataset_key: str, imbalance_ratio: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    selected = learning.loc[
        learning["dataset_key"].eq(dataset_key)
        & learning["imbalance_ratio"].eq(imbalance_ratio)
    ]
    for method, group in selected.groupby("method"):
        group = group.sort_values("training_size")
        ax.plot(group["training_size"], group["roc_auc"], marker="o", label=method)
    ax.set_title(f"{dataset_key}: ROC-AUC learning curve ({imbalance_ratio}:1)")
    ax.set_xlabel("Training size")
    ax.set_ylabel("ROC-AUC")
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig, ax


def plot_mean_learning_curves(learning: pd.DataFrame, *, imbalance_ratio: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    selected = learning.loc[learning["imbalance_ratio"].eq(imbalance_ratio)]
    mean = selected.groupby(["training_size", "method"], as_index=False)["roc_auc"].mean()
    for method, group in mean.groupby("method"):
        group = group.sort_values("training_size")
        ax.plot(group["training_size"], group["roc_auc"], marker="o", label=method)
    ax.set_title(f"Mean ROC-AUC learning curve ({imbalance_ratio}:1)")
    ax.set_xlabel("Training size")
    ax.set_ylabel("ROC-AUC")
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig, ax


def plot_rank_summary(rank: pd.DataFrame, *, segment: str = "full", imbalance_ratio: float | None = None):
    selected = rank.loc[rank["segment"].eq(segment)]
    if imbalance_ratio is not None:
        selected = selected.loc[selected["imbalance_ratio"].eq(imbalance_ratio)]
    summary = selected.groupby("method", as_index=False)["mean_rank"].mean().sort_values("mean_rank")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(summary["method"], summary["mean_rank"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean rank (lower is better)")
    title = f"Method ranks: {segment}"
    if imbalance_ratio is not None:
        title += f" ({imbalance_ratio}:1)"
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def save_all_figures(learning: pd.DataFrame, rank: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    if not learning.empty:
        for (dataset_key, ratio), _group in learning.groupby(["dataset_key", "imbalance_ratio"]):
            fig, _ax = plot_learning_curves(learning, dataset_key=dataset_key, imbalance_ratio=ratio)
            path = output / f"{dataset_key}__ratio-{ratio:g}__learning_curve.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
        for ratio in sorted(learning["imbalance_ratio"].unique()):
            fig, _ax = plot_mean_learning_curves(learning, imbalance_ratio=ratio)
            path = output / f"mean__ratio-{ratio:g}__learning_curve.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    if not rank.empty:
        fig, _ax = plot_rank_summary(rank, segment="full")
        path = output / "rank_summary__full.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
