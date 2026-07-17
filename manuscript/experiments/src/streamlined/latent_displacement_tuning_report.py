"""Readable reports for latent-displacement Bayesian optimization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .latent_displacement_tuning import LatentDisplacementTuningResult


PARAMETER_LABELS = {
    "mimic_capacity": "MIMIC capacity",
    "n_neighbors": "Number of neighbors",
    "lambda_low": "Lambda lower bound",
    "lambda_high": "Lambda upper bound",
}


def top_trials_table(result: LatentDisplacementTuningResult, *, n: int = 10) -> pd.DataFrame:
    """Return the best BO trials with the most relevant columns."""
    history = _history(result)
    columns = [
        "trial",
        "mean_direct_roc_auc",
        "mean_latent_roc_auc",
        "objective_mean_delta_vs_direct",
        "wins_vs_direct",
        "comparisons",
        "mimic_capacity",
        "n_neighbors",
        "lambda_low",
        "lambda_high",
    ]
    available = [column for column in columns if column in history.columns]
    table = history.sort_values("objective_mean_delta_vs_direct", ascending=False).head(max(1, n))[available].copy()
    table["trial"] = table["trial"].astype(int) + 1
    return table.reset_index(drop=True)


def plot_optimization_progress(result: LatentDisplacementTuningResult):
    """Plot trial performance and the best advantage found so far."""
    history = _history(result).sort_values("trial").copy()
    trials = history["trial"].to_numpy() + 1
    delta = history["objective_mean_delta_vs_direct"].to_numpy()
    best = np.maximum.accumulate(delta)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(trials, delta, "o-", alpha=0.65, label="Current trial")
    ax.plot(trials, best, linewidth=2.5, label="Best so far")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, label="Equal to direct")
    ax.set(xlabel="Bayesian-optimization trial", ylabel="Mean ROC-AUC advantage (latent − direct)", title="Optimization progress")
    ax.grid(alpha=0.25)
    ax.legend()
    return fig, ax


def plot_hyperparameter_performance(result: LatentDisplacementTuningResult):
    """Plot a heatmap for every pair of searched hyperparameters."""
    history = _history(result)
    pairs = [
        (left, right)
        for index, left in enumerate(PARAMETER_LABELS)
        for right in list(PARAMETER_LABELS)[index + 1 :]
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    delta = history["objective_mean_delta_vs_direct"].to_numpy(dtype=float)
    color_min = float(np.nanmin(delta))
    color_max = float(np.nanmax(delta))
    if np.isclose(color_min, color_max):
        padding = max(abs(color_min) * 0.05, 1e-6)
        color_min -= padding
        color_max += padding
    best = history.loc[history["objective_mean_delta_vs_direct"].idxmax()]
    heatmap = None
    for ax, (x_column, y_column) in zip(axes.flat, pairs):
        ax.set_facecolor("black")
        surface = _pairwise_surface(history, x_column, y_column)
        heatmap = ax.pcolormesh(
            surface["x_grid"], surface["y_grid"], surface["z_grid"],
            cmap="Blues_r", vmin=color_min, vmax=color_max, shading="auto",
            alpha=surface["alpha_grid"],
        )
        contour_values = np.ma.masked_where(
            surface["alpha_grid"] < 0.15,
            surface["z_grid"],
        )
        ax.contour(
            surface["x_grid"],
            surface["y_grid"],
            contour_values,
            levels=np.linspace(color_min, color_max, 17)[1:-1],
            colors="white",
            linewidths=0.35,
            alpha=0.45,
            zorder=2,
        )
        evaluated = history.drop(index=best.name)
        ax.scatter(
            evaluated[x_column],
            evaluated[y_column],
            s=34,
            facecolor="black",
            edgecolor="white",
            linewidth=0.9,
            alpha=1.0,
            zorder=3,
            label="Evaluated trial",
        )
        ax.scatter(
            best[x_column],
            best[y_column],
            marker="*",
            s=240,
            facecolor="gold",
            edgecolor="black",
            linewidth=1.1,
            alpha=1.0,
            zorder=5,
            label="Selected trial",
        )
        ax.set(xlabel=PARAMETER_LABELS[x_column], ylabel=PARAMETER_LABELS[y_column])
    axes.flat[0].legend(loc="best", fontsize=8)
    if heatmap is not None:
        fig.colorbar(heatmap, ax=axes, label="Mean ROC-AUC advantage (latent − direct)", shrink=0.85)
    fig.suptitle("Pairwise hyperparameter performance heatmaps (smooth supported regions)")
    return fig, axes


def plot_hyperparameter_quantile_bands(
    result: LatentDisplacementTuningResult,
    *,
    n_bins: int = 5,
    loess_fraction: float = 0.65,
):
    """Plot median performance with an interquartile band for each parameter."""
    if not 0 < loess_fraction <= 1:
        raise ValueError("loess_fraction must be in (0, 1]")
    history = _history(result)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharey=True)
    for ax, (column, label) in zip(axes.flat, PARAMETER_LABELS.items()):
        summary = _binned_parameter_summary(history, column, n_bins=n_bins)
        x = summary["parameter_value"].to_numpy(dtype=float)
        q25 = summary["q25"].to_numpy(dtype=float)
        median = summary["median"].to_numpy(dtype=float)
        q75 = summary["q75"].to_numpy(dtype=float)
        ax.fill_between(x, q25, q75, alpha=0.28, color="tab:blue", label="25th–75th percentile")
        ax.plot(x, median, "o-", color="tab:blue", linewidth=1.4, markersize=5, label="Binned median")
        smooth_x, smooth_median = _loess_smooth(x, median, fraction=loess_fraction)
        ax.plot(
            smooth_x,
            smooth_median,
            color="navy",
            linewidth=4,
            solid_capstyle="round",
            label="LOESS-smoothed median",
        )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel=label, ylabel="ROC-AUC advantage (latent − direct)")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(loc="best")
    fig.suptitle("Marginal hyperparameter performance distributions")
    return fig, axes


def _loess_smooth(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fraction: float = 0.65,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a local-linear LOESS curve using tricube distance weights."""
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    if len(x) < 3 or np.isclose(x.min(), x.max()):
        return x, y
    output_x = np.linspace(float(x.min()), float(x.max()), n_points)
    output_y = np.empty_like(output_x)
    neighbor_count = max(2, min(len(x), int(np.ceil(fraction * len(x)))))
    for index, center in enumerate(output_x):
        distance = np.abs(x - center)
        bandwidth = np.partition(distance, neighbor_count - 1)[neighbor_count - 1]
        bandwidth = max(float(bandwidth), np.finfo(float).eps)
        scaled = np.clip(distance / bandwidth, 0.0, 1.0)
        weights = (1.0 - scaled**3) ** 3
        design = np.column_stack((np.ones(len(x)), x - center))
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_y = y * np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
        output_y[index] = coefficients[0]
    return output_x, output_y


def _binned_parameter_summary(history: pd.DataFrame, column: str, *, n_bins: int) -> pd.DataFrame:
    values = history[column]
    unique = values.nunique()
    if unique <= max(2, n_bins):
        groups = values
    else:
        groups = pd.qcut(values, q=min(n_bins, unique), duplicates="drop")
    frame = history.assign(_group=groups)
    summary = (
        frame.groupby("_group", observed=True)["objective_mean_delta_vs_direct"]
        .agg(q25=lambda x: x.quantile(0.25), median="median", q75=lambda x: x.quantile(0.75))
        .reset_index()
    )
    if isinstance(summary["_group"].dtype, pd.CategoricalDtype):
        summary["parameter_value"] = summary["_group"].map(lambda interval: interval.mid).astype(float)
    else:
        summary["parameter_value"] = summary["_group"].astype(float)
    return summary.sort_values("parameter_value").reset_index(drop=True)


def _pairwise_surface(
    history: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    bandwidth: float = 0.095,
    full_support_radius: float = 0.22,
    zero_support_radius: float = 0.38,
) -> dict[str, np.ndarray]:
    grouped = history.groupby([x_column, y_column], as_index=False)["objective_mean_delta_vs_direct"].mean()
    x = grouped[x_column].to_numpy(dtype=float)
    y = grouped[y_column].to_numpy(dtype=float)
    z = grouped["objective_mean_delta_vs_direct"].to_numpy(dtype=float)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_padding = _axis_padding(x_min, x_max)
    y_padding = _axis_padding(y_min, y_max)
    x_values = np.linspace(x_min - x_padding, x_max + x_padding, 60)
    y_values = np.linspace(y_min - y_padding, y_max + y_padding, 60)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    x_span = max(x_max - x_min, np.finfo(float).eps)
    y_span = max(y_max - y_min, np.finfo(float).eps)
    training_points = np.column_stack(((x - x_min) / x_span, (y - y_min) / y_span))
    prediction_points = np.column_stack(
        (((x_grid.ravel() - x_min) / x_span), ((y_grid.ravel() - y_min) / y_span))
    )
    squared_distances = np.sum(
        (prediction_points[:, None, :] - training_points[None, :, :]) ** 2,
        axis=2,
    )
    weights = np.exp(-0.5 * squared_distances / max(bandwidth**2, np.finfo(float).eps))
    z_grid = ((weights @ z) / np.maximum(weights.sum(axis=1), np.finfo(float).eps)).reshape(x_grid.shape)
    nearest_distance = np.sqrt(np.min(squared_distances, axis=1)).reshape(x_grid.shape)
    fade = np.clip(
        (zero_support_radius - nearest_distance) / (zero_support_radius - full_support_radius),
        0.0,
        1.0,
    )
    alpha_grid = fade * fade * (3.0 - 2.0 * fade)
    return {"x_grid": x_grid, "y_grid": y_grid, "z_grid": z_grid, "alpha_grid": alpha_grid}


def _axis_padding(low: float, high: float, *, fraction: float = 0.07) -> float:
    span = high - low
    if span > 0:
        return span * fraction
    return max(abs(low) * fraction, 0.1)


def heldout_conclusion(result: LatentDisplacementTuningResult) -> pd.DataFrame:
    """Build a one-row plain-language conclusion from the untouched test results."""
    summary = result.heldout_summary
    columns = [
        "verdict",
        "direct_mean_roc_auc",
        "latent_mean_roc_auc",
        "mean_roc_auc_advantage",
        "wins",
        "comparisons",
        "selected_hyperparameters",
    ]
    if summary.empty or "method" not in summary.columns:
        return pd.DataFrame(columns=columns)
    latent = summary.loc[summary["method"].eq("latent_displacement")]
    direct = summary.loc[summary["method"].eq("direct_displacement")]
    if latent.empty or direct.empty:
        return pd.DataFrame(columns=columns)
    latent_row, direct_row = latent.iloc[0], direct.iloc[0]
    delta = float(latent_row["mean_delta_vs_direct"])
    verdict = "better" if delta > 0 else "worse" if delta < 0 else "equal"
    return pd.DataFrame(
        [{
            "verdict": f"Tuned latent displacement was {verdict} than direct displacement on held-out data.",
            "direct_mean_roc_auc": direct_row["mean_roc_auc"],
            "latent_mean_roc_auc": latent_row["mean_roc_auc"],
            "mean_roc_auc_advantage": delta,
            "wins": int(latent_row["wins_vs_direct"]),
            "comparisons": int(latent_row["comparisons"]),
            "selected_hyperparameters": result.selected_candidate.candidate_id,
        }]
    )


def save_optimization_report(result: LatentDisplacementTuningResult, artifact_dir: str | Path) -> pd.DataFrame:
    """Save report tables and plots and return their paths."""
    root = Path(artifact_dir) / "tuning" / "latent_displacement_default_credit" / "report"
    root.mkdir(parents=True, exist_ok=True)
    top_path = root / "top_trials.csv"
    conclusion_path = root / "heldout_conclusion.csv"
    top_trials_table(result).to_csv(top_path, index=False)
    heldout_conclusion(result).to_csv(conclusion_path, index=False)
    progress, _ = plot_optimization_progress(result)
    parameter_plot, _ = plot_hyperparameter_performance(result)
    quantile_plot, _ = plot_hyperparameter_quantile_bands(result)
    progress_path = root / "optimization_progress.png"
    parameter_path = root / "hyperparameter_performance.png"
    quantile_path = root / "hyperparameter_quantile_bands.png"
    progress.savefig(progress_path, dpi=180, bbox_inches="tight")
    parameter_plot.savefig(parameter_path, dpi=180, bbox_inches="tight")
    quantile_plot.savefig(quantile_path, dpi=180, bbox_inches="tight")
    plt.close(progress)
    plt.close(parameter_plot)
    plt.close(quantile_plot)
    return pd.DataFrame(
        {"artifact": ["top_trials", "heldout_conclusion", "optimization_progress", "hyperparameter_performance", "hyperparameter_quantile_bands"],
         "path": [str(top_path), str(conclusion_path), str(progress_path), str(parameter_path), str(quantile_path)]}
    )


def _history(result: LatentDisplacementTuningResult) -> pd.DataFrame:
    if result.optimization_history is None or result.optimization_history.empty:
        raise ValueError("The result has no Bayesian-optimization history")
    return result.optimization_history
