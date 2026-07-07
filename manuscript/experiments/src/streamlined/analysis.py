"""Analysis tables for streamlined experiment outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import METHOD_KEYS, ExperimentConfig


def learning_curves(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    group = ["dataset_key", "imbalance_ratio", "training_size", "method"]
    metrics = ["roc_auc", "pr_auc", "balanced_accuracy", "f1", "brier"]
    return results.groupby(group, as_index=False)[metrics].mean()


def aulc_table(results: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    if results.empty:
        return pd.DataFrame(columns=["dataset_key", "imbalance_ratio", "method", "seed", "segment", "aulc"])
    for keys, group in results.groupby(["dataset_key", "imbalance_ratio", "method", "seed"]):
        dataset_key, ratio, method, seed = keys
        for segment, bounds in _segments(config).items():
            selected = _segment_rows(group, bounds)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "imbalance_ratio": ratio,
                    "method": method,
                    "seed": seed,
                    "segment": segment,
                    "aulc": _aulc(selected["training_size"].to_numpy(), selected["roc_auc"].to_numpy()),
                }
            )
    return pd.DataFrame(rows)


def pairwise_comparisons(aulc: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    pairs = [
        ("latent_displacement", "latent_smote"),
        ("direct_displacement", "direct_smote"),
        ("latent_smote", "direct_smote"),
        ("latent_displacement", "direct_displacement"),
        ("direct_smote", "real_balanced"),
        ("direct_displacement", "real_balanced"),
        ("latent_smote", "real_balanced"),
        ("latent_displacement", "real_balanced"),
    ]
    rows = []
    for keys, group in aulc.groupby(["dataset_key", "imbalance_ratio", "segment"]):
        dataset_key, ratio, segment = keys
        pivot = group.pivot_table(index="seed", columns="method", values="aulc")
        for left, right in pairs:
            if left not in pivot or right not in pivot:
                continue
            diff = (pivot[left] - pivot[right]).dropna()
            if diff.empty:
                continue
            low, high = _bootstrap_ci(diff.to_numpy())
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "imbalance_ratio": ratio,
                    "segment": segment,
                    "left_method": left,
                    "right_method": right,
                    "mean_delta": float(diff.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "n": int(len(diff)),
                    "indistinguishable": bool(abs(float(diff.mean())) <= config.equivalence_margin),
                }
            )
    return pd.DataFrame(rows)


def rank_summary(aulc: pd.DataFrame) -> pd.DataFrame:
    if aulc.empty:
        return pd.DataFrame(columns=["segment", "imbalance_ratio", "method", "mean_rank"])
    rows = []
    grouped = aulc.groupby(["dataset_key", "imbalance_ratio", "segment", "method"], as_index=False)["aulc"].mean()
    for keys, group in grouped.groupby(["imbalance_ratio", "segment"]):
        ratio, segment = keys
        ranks = group.assign(rank=group.groupby("dataset_key")["aulc"].rank(ascending=False, method="average"))
        for method, method_group in ranks.groupby("method"):
            rows.append({"segment": segment, "imbalance_ratio": ratio, "method": method, "mean_rank": float(method_group["rank"].mean())})
    return pd.DataFrame(rows)


def regime_summary(aulc: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if aulc.empty:
        return pd.DataFrame(columns=["method", "best_segment", "best_imbalance_ratio", "mean_full_aulc", "indistinguishable_from_real"])
    full = aulc.loc[aulc["segment"].eq("full")]
    method_scores = full.groupby(["method", "imbalance_ratio"], as_index=False)["aulc"].mean()
    for method in METHOD_KEYS:
        selected = method_scores.loc[method_scores["method"].eq(method)]
        if selected.empty:
            continue
        best = selected.sort_values("aulc", ascending=False).iloc[0]
        real_pairs = pairwise.loc[
            pairwise["left_method"].eq(method)
            & pairwise["right_method"].eq("real_balanced")
            & pairwise["indistinguishable"].eq(True)
        ]
        rows.append(
            {
                "method": method,
                "best_segment": _best_segment(aulc, method),
                "best_imbalance_ratio": float(best["imbalance_ratio"]),
                "mean_full_aulc": float(selected["aulc"].mean()),
                "indistinguishable_from_real": bool(not real_pairs.empty),
            }
        )
    return pd.DataFrame(rows)


def prescriptive_conclusions(regime: pd.DataFrame) -> str:
    lines = ["# Prescriptive Conclusions", ""]
    if regime.empty:
        lines.append("No completed results are available yet.")
        return "\n".join(lines) + "\n"
    best = regime.sort_values("mean_full_aulc", ascending=False).iloc[0]
    lines.append(f"- Use `{best['method']}` as the current strongest overall method in completed runs.")
    for _, row in regime.iterrows():
        status = "is" if row["indistinguishable_from_real"] else "is not"
        lines.append(
            f"- `{row['method']}` performs best around imbalance `{row['best_imbalance_ratio']}:1` "
            f"in the `{row['best_segment']}` regime and {status} indistinguishable from real balanced data in at least one completed comparison."
        )
    return "\n".join(lines) + "\n"


def _segments(config: ExperimentConfig) -> dict[str, tuple[int | None, int | None]]:
    if config.aulc_segments:
        return {"full": (None, None), **config.aulc_segments}
    sizes = sorted(config.profile.training_sizes)
    if len(sizes) < 3:
        return {"full": (None, None), "early": (None, sizes[0]), "mid": (sizes[0], None)}
    return {"full": (None, None), "early": (None, sizes[1]), "mid": (sizes[1], sizes[-1])}


def _segment_rows(group: pd.DataFrame, bounds: tuple[int | None, int | None]) -> pd.DataFrame:
    start, end = bounds
    selected = group
    if start is not None:
        selected = selected.loc[selected["training_size"] >= start]
    if end is not None:
        selected = selected.loc[selected["training_size"] <= end]
    return selected.sort_values("training_size")


def _aulc(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    if len(x) == 0:
        return float("nan")
    if len(x) == 1:
        return float(y[0])
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]) / (x[order][-1] - x[order][0]))


def _bootstrap_ci(values: np.ndarray, *, random_state: int = 0, n_resamples: int = 500) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(random_state)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_resamples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _best_segment(aulc: pd.DataFrame, method: str) -> str:
    selected = aulc.loc[aulc["method"].eq(method)]
    if selected.empty:
        return "unknown"
    scores = selected.groupby("segment")["aulc"].mean()
    return str(scores.idxmax())
