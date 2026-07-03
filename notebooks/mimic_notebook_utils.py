"""Reusable helpers for MIMIC example notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

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
    ax.scatter(majority["x"], majority["y"], label="original majority", color="#9aa0a6", alpha=0.32, s=24)
    ax.scatter(minority["x"], minority["y"], label="original minority", color="#1f77b4", alpha=0.82, s=36)
    ax.scatter(samples["x"], samples["y"], label="generated minority", color="#ff7f0e", marker="x", alpha=0.9, s=44)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    return summary, samples, trace, fig, ax
