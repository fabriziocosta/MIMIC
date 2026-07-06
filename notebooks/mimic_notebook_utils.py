"""Reusable helpers for MIMIC example notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mimic.diagnostics import classical_mds_2d

GENERATED_COLOR = "#ff2e0e"
ORIGINAL_MAJORITY_COLOR = "#9aa0a6"
ORIGINAL_MINORITY_COLOR = "#1f77b4"


def make_two_spiral_frame(
    majority_samples: int = 180,
    minority_samples: int = 90,
    random_state: int = 2,
) -> pd.DataFrame:
    """Create the undersampled two-spiral demo frame used by generation notebooks."""

    if minority_samples > majority_samples:
        raise ValueError("minority_samples cannot exceed majority_samples")

    rng = np.random.default_rng(random_state)

    def make_spiral(label: str, n: int, phase: float, noise: float = 0.12) -> pd.DataFrame:
        theta = np.linspace(0.45, 4.8 * np.pi, n)
        radius = np.linspace(0.25, 4.0, n)
        x = radius * np.cos(theta + phase) + rng.normal(0, noise, n)
        y = radius * np.sin(theta + phase) + rng.normal(0, noise, n)
        return pd.DataFrame({"x": x, "y": y, "label": label})

    majority = make_spiral("majority", majority_samples, phase=0.0)
    minority_full = make_spiral("minority", majority_samples, phase=np.pi)
    minority = minority_full.sample(n=minority_samples, random_state=random_state).sort_index().reset_index(drop=True)
    df = pd.concat([majority, minority], ignore_index=True)
    df["id"] = np.arange(len(df))
    return df


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready class balance table."""

    return df["label"].value_counts().rename_axis("label").to_frame("count")


def append_generated_rows(df: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:
    """Combine original rows and generated rows for balance summaries."""

    return pd.concat([df.drop(columns=["id"], errors="ignore"), generated], ignore_index=True)


def generated_embedding_trace(samples: pd.DataFrame, trace: pd.DataFrame, include_condition: bool = False) -> pd.DataFrame:
    """Join generated rows to their embedding-generation trace columns."""

    trace_columns = [
        "anchor_index",
        "displacement_from_index",
        "displacement_to_index",
        "lambda",
        "neighbour_mode",
        "decoder",
    ]
    if include_condition:
        trace_columns.insert(-1, "condition")

    embedding_trace = trace.loc[trace["trace_type"].eq("embedding")].set_index("sample_index")
    return samples.join(embedding_trace[trace_columns])


def cell_sampling_trace(trace: pd.DataFrame) -> pd.DataFrame:
    """Return only per-cell sampling trace columns with empty columns removed."""

    return trace.loc[trace["trace_type"].eq("cell")].dropna(axis=1, how="all")


def plot_oversampling(df: pd.DataFrame, generated: pd.DataFrame, title: str = "Displacement oversampling"):
    """Plot original/generated oversampling views in one row."""

    plot_df = pd.concat(
        [
            df.drop(columns=["id"], errors="ignore").assign(source="original"),
            generated.assign(source="generated"),
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)

    panels = [
        (plot_df, title),
        (plot_df[plot_df["source"].eq("original")], "Original majority and minority"),
        (
            plot_df[
                (plot_df["source"].eq("generated") & plot_df["label"].eq("minority"))
                | (plot_df["source"].eq("original") & plot_df["label"].eq("majority"))
            ],
            "Original majority and generated minority",
        ),
    ]

    for ax, (panel_df, panel_title) in zip(axes, panels):
        for (source, label), part in panel_df.groupby(["source", "label"], sort=False):
            if source == "original" and label == "majority":
                color, alpha, size = ORIGINAL_MAJORITY_COLOR, 0.42, 28
            elif source == "original" and label == "minority":
                color, alpha, size = ORIGINAL_MINORITY_COLOR, 0.9, 46
            else:
                color, alpha, size = GENERATED_COLOR, 0.9, 52
            ax.scatter(part["x"], part["y"], label=f"{source} {label}", color=color, marker="o", alpha=alpha, s=size)
        ax.set_title(panel_title)
        ax.set_xlabel("x")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, fontsize="small")
    axes[0].set_ylabel("y")
    fig.tight_layout()
    return fig, axes


def identity_generation_plot(
    model,
    df: pd.DataFrame,
    n_samples: int,
    condition: dict[str, object],
    title: str = "Identity-space generation",
):
    """Fit a user-configured model, sample conditioned rows, and plot them."""

    fitted = model.fit(df)
    samples, trace = fitted.sample(n_samples, condition=condition, return_trace=True)
    policy = fitted.policy_
    summary = pd.DataFrame(
        [
            {
                "method": policy.method,
                "neighbour_mode": policy.neighbour_mode,
                "n_neighbors": policy.n_neighbors,
                "lambda_range": policy.lambda_range,
                "n_generated": len(samples),
                "trace_rows": len(trace),
                "x_min": samples["x"].min(),
                "x_max": samples["x"].max(),
                "y_min": samples["y"].min(),
                "y_max": samples["y"].max(),
            }
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    base = df.drop(columns=["id"], errors="ignore")
    majority = base[base["label"] == "majority"]
    minority = base[base["label"] == "minority"]
    ax.scatter(majority["x"], majority["y"], label="original majority", color=ORIGINAL_MAJORITY_COLOR, alpha=0.32, s=24)
    ax.scatter(minority["x"], minority["y"], label="original minority", color=ORIGINAL_MINORITY_COLOR, alpha=0.82, s=36)
    ax.scatter(samples["x"], samples["y"], label="generated minority", color=GENERATED_COLOR, marker="o", alpha=0.9, s=52)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    return summary, samples, trace, fig, ax


def plot_feature_embedding_grid(
    model,
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    n_cols: int = 4,
    random_state=None,
    point_size: int = 12,
    alpha: float = 0.75,
):
    """Plot one 2D projection per feature embedding, coloured by that feature's raw value."""

    columns = list(model.model_columns_ if columns is None else columns)
    if not columns:
        raise ValueError("columns must contain at least one feature")

    H = model.transform(df)
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.7 * n_rows), squeeze=False)
    flat_axes = axes.ravel()

    for ax, column in zip(flat_axes, columns):
        if column not in model.embedding_slices_:
            raise ValueError(f"Unknown embedding column {column!r}")
        sl = model.embedding_slices_[column]
        coords = classical_mds_2d(H[:, sl], random_state=random_state)
        values = df[column]
        if pd.api.types.is_numeric_dtype(values):
            scatter = ax.scatter(
                coords["mds1"],
                coords["mds2"],
                c=pd.to_numeric(values, errors="coerce"),
                s=point_size,
                alpha=alpha,
                cmap="viridis",
                edgecolor="none",
            )
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
        else:
            labels = values.astype("object").where(values.notna(), "__missing__")
            codes, uniques = pd.factorize(labels)
            ax.scatter(
                coords["mds1"],
                coords["mds2"],
                c=codes,
                s=point_size,
                alpha=alpha,
                cmap="tab20",
                edgecolor="none",
            )
            ax.text(
                0.02,
                0.02,
                f"{len(uniques)} categories",
                transform=ax.transAxes,
                fontsize="x-small",
                alpha=0.7,
            )
        ax.set_title(column, fontsize="medium")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in flat_axes[len(columns) :]:
        ax.set_axis_off()

    fig.tight_layout()
    return fig, axes


def pareto_feature_pair_report(
    df: pd.DataFrame,
    target: str,
    columns: list[str] | None = None,
    *,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Rank feature pairs by high target association and low inter-feature association."""

    columns = [c for c in (df.columns if columns is None else columns) if c != target]
    target_scores = {column: _feature_association(df[column], df[target]) for column in columns}
    rows = []
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            target_association = float((target_scores[left] + target_scores[right]) / 2)
            pair_association = float(_feature_association(df[left], df[right]))
            rows.append(
                {
                    "feature_x": left,
                    "feature_y": right,
                    "target_association": target_association,
                    "pair_association": pair_association,
                    "pareto": False,
                }
            )
    report = pd.DataFrame(rows)
    if report.empty:
        return report

    scores = report[["target_association", "pair_association"]].to_numpy()
    pareto = np.ones(len(report), dtype=bool)
    for i, (target_score, pair_score) in enumerate(scores):
        dominates = (
            (scores[:, 0] >= target_score)
            & (scores[:, 1] <= pair_score)
            & ((scores[:, 0] > target_score) | (scores[:, 1] < pair_score))
        )
        pareto[i] = not dominates.any()
    report["pareto"] = pareto
    report = report.sort_values(["pareto", "target_association", "pair_association"], ascending=[False, False, True])
    if max_pairs is not None:
        pareto_part = report[report["pareto"]].head(max_pairs)
        non_pareto_part = report[~report["pareto"]]
        report = pd.concat([pareto_part, non_pareto_part], ignore_index=True)
    return report.reset_index(drop=True)


def plot_pareto_feature_pairs(
    df: pd.DataFrame,
    target: str,
    columns: list[str] | None = None,
    *,
    max_pairs: int = 6,
):
    """Plot Pareto-optimal feature pairs coloured by the target column."""

    report = pareto_feature_pair_report(df, target, columns=columns, max_pairs=max_pairs)
    front = report[report["pareto"]].head(max_pairs)
    if front.empty:
        raise ValueError("No feature pairs are available for Pareto plotting")

    n_cols = min(3, len(front))
    n_rows = int(np.ceil(len(front) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.4 * n_rows), squeeze=False)
    flat_axes = axes.ravel()
    target_values = df[target]
    target_is_numeric = pd.api.types.is_numeric_dtype(target_values)
    if target_is_numeric:
        color_values = pd.to_numeric(target_values, errors="coerce")
        uniques = None
    else:
        labels = target_values.astype("object").where(target_values.notna(), "__missing__")
        color_values, uniques = pd.factorize(labels)

    for ax, row in zip(flat_axes, front.itertuples(index=False)):
        x = _plot_axis_values(df[row.feature_x])
        y = _plot_axis_values(df[row.feature_y])
        scatter = ax.scatter(x, y, c=color_values, cmap="viridis" if target_is_numeric else "tab10", s=16, alpha=0.72, edgecolor="none")
        ax.set_xlabel(row.feature_x)
        ax.set_ylabel(row.feature_y)
        ax.set_title(f"target={row.target_association:.2f}, pair={row.pair_association:.2f}", fontsize="small")
        if not pd.api.types.is_numeric_dtype(df[row.feature_x]):
            ax.set_xticks(sorted(pd.unique(x)))
        if not pd.api.types.is_numeric_dtype(df[row.feature_y]):
            ax.set_yticks(sorted(pd.unique(y)))

    for ax in flat_axes[len(front) :]:
        ax.set_axis_off()

    if target_is_numeric:
        fig.colorbar(scatter, ax=flat_axes[: len(front)], label=target, fraction=0.025, pad=0.02)
    elif uniques is not None:
        handles = scatter.legend_elements()[0]
        fig.legend(handles, [str(u) for u in uniques], title=target, loc="center right", frameon=False)
        fig.subplots_adjust(right=0.86)
    fig.tight_layout()
    return fig, axes, report


def _plot_axis_values(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    codes, _uniques = pd.factorize(series.astype("object").where(series.notna(), "__missing__"))
    return codes


def _feature_association(left: pd.Series, right: pd.Series) -> float:
    left = pd.Series(left)
    right = pd.Series(right)
    mask = left.notna() & right.notna()
    if mask.sum() < 2:
        return 0.0
    left = left[mask]
    right = right[mask]
    left_numeric = pd.api.types.is_numeric_dtype(left)
    right_numeric = pd.api.types.is_numeric_dtype(right)
    if left_numeric and right_numeric:
        corr = pd.to_numeric(left, errors="coerce").corr(pd.to_numeric(right, errors="coerce"))
        return 0.0 if pd.isna(corr) else float(abs(corr))
    if left_numeric != right_numeric:
        numeric = pd.to_numeric(left if left_numeric else right, errors="coerce")
        categorical = right if left_numeric else left
        return _correlation_ratio(categorical, numeric)
    return _cramers_v(left, right)


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    frame = pd.DataFrame({"category": categories.astype("object"), "value": values}).dropna()
    if frame.empty:
        return 0.0
    grand_mean = frame["value"].mean()
    total = ((frame["value"] - grand_mean) ** 2).sum()
    if total <= 0:
        return 0.0
    between = frame.groupby("category")["value"].agg(lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()
    return float(np.sqrt(max(0.0, between / total)))


def _cramers_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(left.astype("object"), right.astype("object"))
    n = table.to_numpy().sum()
    if n == 0:
        return 0.0
    expected = np.outer(table.sum(axis=1).to_numpy(), table.sum(axis=0).to_numpy()) / n
    observed = table.to_numpy()
    valid = expected > 0
    chi2 = (((observed - expected) ** 2) / np.where(valid, expected, 1.0))[valid].sum()
    denom = n * max(1, min(table.shape) - 1)
    return 0.0 if denom == 0 else float(np.sqrt(chi2 / denom))
