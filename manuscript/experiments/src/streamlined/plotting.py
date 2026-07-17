"""Plotting helpers for streamlined experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp

from .analysis import real_equivalent_sample_fraction


REAL_BASELINE_METHOD = "real_balanced"


def plot_learning_curves(learning: pd.DataFrame, *, dataset_key: str, imbalance_ratio: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    selected = learning.loc[
        learning["dataset_key"].eq(dataset_key)
        & learning["imbalance_ratio"].eq(imbalance_ratio)
    ]
    equivalent_fractions = _mean_real_equivalent_fractions(selected)
    for method, group in selected.groupby("method"):
        group = group.sort_values("training_size")
        ax.errorbar(
            group["training_size"],
            group["roc_auc"],
            yerr=_roc_auc_std(group),
            marker="o",
            capsize=3,
            label=_learning_curve_label(method, equivalent_fractions),
        )
    ax.set_title(f"{dataset_key}: ROC-AUC learning curve ({imbalance_ratio}:1)")
    ax.set_xlabel("Training size")
    ax.set_ylabel("ROC-AUC")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig, ax


def generate_mean_learning_curves(learning: pd.DataFrame, *, imbalance_ratio: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    selected = learning.loc[learning["imbalance_ratio"].eq(imbalance_ratio)]
    aggregations = {"roc_auc": "mean"}
    if "roc_auc_std" in selected.columns:
        aggregations["roc_auc_std"] = _root_mean_square
    mean = selected.groupby(["training_size", "method"], as_index=False).agg(aggregations)
    equivalent_fractions = _mean_real_equivalent_fractions(selected)
    for method, group in mean.groupby("method"):
        group = group.sort_values("training_size")
        ax.errorbar(
            group["training_size"],
            group["roc_auc"],
            yerr=_roc_auc_std(group),
            marker="o",
            capsize=3,
            label=_learning_curve_label(method, equivalent_fractions),
        )
    ax.set_title(f"Mean ROC-AUC learning curve ({imbalance_ratio}:1)")
    ax.set_xlabel("Training size")
    ax.set_ylabel("ROC-AUC")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig, ax


def plot_real_equivalent_by_dataset(
    equivalence: pd.DataFrame,
    *,
    imbalance_ratio: float | None = None,
):
    """Plot mean real-equivalent fractions grouped by dataset and method."""
    selected = equivalence
    if imbalance_ratio is not None:
        selected = selected.loc[selected["imbalance_ratio"].eq(imbalance_ratio)]
    grouped = selected.groupby(["dataset_key", "method"])["real_equivalent_fraction"]
    summary = grouped.mean().unstack("method")
    errors = grouped.std().unstack("method").reindex_like(summary).fillna(0.0)
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(summary.index)), 4))
    if summary.empty:
        ax.set_axis_off()
        ax.set_title("Real-equivalent fraction: no data")
        return fig, ax
    summary.plot.bar(ax=ax, yerr=errors, capsize=3)
    title = "Mean real-equivalent fraction by dataset and method"
    if imbalance_ratio is not None:
        title += f" ({imbalance_ratio}:1)"
    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mean real-equivalent fraction (±1 SD across training sizes)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(
        title="Method",
        fontsize="small",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=max(1, len(summary.columns)),
    )
    fig.tight_layout()
    return fig, ax


def plot_rank_summary(rank: pd.DataFrame, *, segment: str = "full", imbalance_ratio: float | None = None):
    selected = rank.loc[rank["segment"].eq(segment)]
    if imbalance_ratio is not None:
        selected = selected.loc[selected["imbalance_ratio"].eq(imbalance_ratio)]
    summary = selected.groupby("method", as_index=False)["mean_rank"].mean().sort_values("mean_rank")
    return _plot_bar_rank_summary(summary, segment=segment, imbalance_ratio=imbalance_ratio)


def generate_critical_difference_diagram(aulc: pd.DataFrame, *, segment: str = "full", imbalance_ratio: float | None = None):
    selected = aulc.loc[aulc["segment"].eq(segment)]
    if imbalance_ratio is not None:
        selected = selected.loc[selected["imbalance_ratio"].eq(imbalance_ratio)]
    ranks, sig_matrix = critical_difference_inputs(selected)
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(ranks) < 2:
        ax.set_axis_off()
        ax.set_title("Critical difference diagram: not enough methods")
        return fig, ax
    sp.critical_difference_diagram(
        ranks,
        sig_matrix,
        ax=ax,
        label_props={"fontsize": 8},
        marker_props={"s": 45},
    )
    title = f"Critical difference diagram: {segment}"
    if imbalance_ratio is not None:
        title += f" ({imbalance_ratio}:1)"
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def critical_difference_inputs(selected_aulc: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    if selected_aulc.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    mean_by_block = selected_aulc.groupby(["dataset_key", "seed", "method"], as_index=False)["aulc"].mean()
    score_matrix = mean_by_block.pivot_table(index=["dataset_key", "seed"], columns="method", values="aulc")
    score_matrix = score_matrix.dropna(axis=0, how="any")
    if score_matrix.empty:
        methods = sorted(selected_aulc["method"].unique())
        ranks = pd.Series({method: np.nan for method in methods}, dtype=float).dropna()
        return ranks, pd.DataFrame(1.0, index=methods, columns=methods)
    rank_matrix = score_matrix.rank(axis=1, ascending=False, method="average")
    ranks = rank_matrix.mean(axis=0).sort_values()
    sig_matrix = _nemenyi_sig_matrix(score_matrix[ranks.index])
    return ranks, sig_matrix


def plot_mean_learning_curves(learning: pd.DataFrame, *, imbalance_ratio: float):
    return generate_mean_learning_curves(learning, imbalance_ratio=imbalance_ratio)


def _mean_real_equivalent_fractions(learning: pd.DataFrame) -> dict[str, float]:
    equivalence = real_equivalent_sample_fraction(
        learning,
        baseline_method=REAL_BASELINE_METHOD,
    )
    fractions = equivalence.groupby("method")["real_equivalent_fraction"].mean().to_dict()
    if REAL_BASELINE_METHOD in learning["method"].values:
        fractions[REAL_BASELINE_METHOD] = 1.0
    return fractions


def _learning_curve_label(method: str, equivalent_fractions: dict[str, float]) -> str:
    fraction = equivalent_fractions.get(method)
    formatted_fraction = "n/a" if fraction is None or not np.isfinite(fraction) else f"{fraction:.2f}"
    return f"{method} (real equivalent: {formatted_fraction})"


def _roc_auc_std(group: pd.DataFrame):
    if "roc_auc_std" not in group.columns:
        return None
    return group["roc_auc_std"].fillna(0.0)


def _root_mean_square(values: pd.Series) -> float:
    finite = values.dropna().to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(finite)))) if len(finite) else np.nan


def plot_critical_difference_diagram(aulc: pd.DataFrame, *, segment: str = "full", imbalance_ratio: float | None = None):
    return generate_critical_difference_diagram(aulc, segment=segment, imbalance_ratio=imbalance_ratio)


def _plot_bar_rank_summary(summary: pd.DataFrame, *, segment: str, imbalance_ratio: float | None):
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


def _nemenyi_sig_matrix(score_matrix: pd.DataFrame) -> pd.DataFrame:
    methods = list(score_matrix.columns)
    if len(methods) < 2 or len(score_matrix) < 2:
        return pd.DataFrame(1.0, index=methods, columns=methods)
    block_matrix = score_matrix.copy()
    block_matrix.index = [f"{dataset_key}__seed-{seed}" for dataset_key, seed in block_matrix.index]
    block_matrix.index.name = "block"
    melted = block_matrix.reset_index().melt(id_vars="block", var_name="method", value_name="score")
    try:
        matrix = sp.posthoc_nemenyi_friedman(melted, y_col="score", block_col="block", group_col="method", melted=True)
        return matrix.loc[methods, methods]
    except Exception:
        return pd.DataFrame(1.0, index=methods, columns=methods)


def save_all_figures(learning: pd.DataFrame, rank: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    if not learning.empty:
        for (dataset_key, ratio), _group in learning.groupby(["dataset_key", "imbalance_ratio"]):
            fig, _ax = plot_learning_curves(learning, dataset_key=dataset_key, imbalance_ratio=ratio)
            path = output / f"{dataset_key}__ratio-{ratio:g}__learning_curve"
            paths.extend(_save_figure(fig, path))
            plt.close(fig)
        for ratio in sorted(learning["imbalance_ratio"].unique()):
            fig, _ax = generate_mean_learning_curves(learning, imbalance_ratio=ratio)
            path = output / f"mean__ratio-{ratio:g}__learning_curve"
            paths.extend(_save_figure(fig, path))
            plt.close(fig)
    if not rank.empty:
        fig, _ax = plot_rank_summary(rank, segment="full")
        path = output / "rank_summary__full"
        paths.extend(_save_figure(fig, path))
        plt.close(fig)
    # Critical-difference diagrams need per-block AULC values, so callers should
    # save them separately through save_critical_difference_figures.
    return paths


def save_critical_difference_figures(aulc: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    if aulc.empty:
        return paths
    for segment in sorted(aulc["segment"].dropna().unique()):
        fig, _ax = generate_critical_difference_diagram(aulc, segment=segment)
        path = output / f"critical_difference__{segment}"
        paths.extend(_save_figure(fig, path))
        plt.close(fig)
        for ratio in sorted(aulc["imbalance_ratio"].dropna().unique()):
            fig, _ax = generate_critical_difference_diagram(aulc, segment=segment, imbalance_ratio=ratio)
            path = output / f"critical_difference__{segment}__ratio-{ratio:g}"
            paths.extend(_save_figure(fig, path))
            plt.close(fig)
    return paths


def _save_figure(fig, stem: Path) -> list[Path]:
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=150)
    fig.savefig(svg)
    return [png, svg]
