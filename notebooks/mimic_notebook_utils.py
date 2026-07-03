"""Reusable helpers for MIMIC example notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
