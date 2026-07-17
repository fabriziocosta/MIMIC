"""Validation-led hyperparameter tuning for latent displacement."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
import json

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skopt import Optimizer
from skopt.space import Integer, Real

from .config import ExperimentConfig
from .datasets import load_dataset
from .preprocessing import fit_preprocess_train_test
from .runner import run_condition
from .sampling import make_imbalanced_subset


@dataclass(frozen=True)
class LatentDisplacementCandidate:
    mimic_capacity: float
    n_neighbors: int
    lambda_range: tuple[float, float]
    mimic_mode: str = "factorised"

    @property
    def candidate_id(self) -> str:
        low, high = self.lambda_range
        return (
            f"mode={self.mimic_mode}|capacity={self.mimic_capacity:g}|"
            f"neighbors={self.n_neighbors}|lambda={low:g}:{high:g}"
        )


@dataclass
class LatentDisplacementTuningResult:
    validation_results: pd.DataFrame
    validation_summary: pd.DataFrame
    selected_candidate: LatentDisplacementCandidate
    heldout_results: pd.DataFrame
    heldout_summary: pd.DataFrame
    optimization_history: pd.DataFrame | None = None


@dataclass(frozen=True)
class BayesianOptimizationSpace:
    capacity_bounds: tuple[float, float] = (0.05, 0.8)
    neighbor_bounds: tuple[int, int] = (2, 15)
    lambda_low_bounds: tuple[float, float] = (0.0, 0.5)
    lambda_width_bounds: tuple[float, float] = (0.1, 1.5)
    mimic_mode: str = "factorised"


def candidate_grid(
    *,
    capacities: tuple[float, ...],
    neighbor_counts: tuple[int, ...],
    lambda_ranges: tuple[tuple[float, float], ...],
    modes: tuple[str, ...] = ("factorised",),
) -> tuple[LatentDisplacementCandidate, ...]:
    """Return a deterministic Cartesian product of tuning candidates."""
    return tuple(
        LatentDisplacementCandidate(capacity, neighbors, lambda_range, mode)
        for mode, capacity, neighbors, lambda_range in product(
            modes, capacities, neighbor_counts, lambda_ranges
        )
    )


def select_candidate(validation_results: pd.DataFrame) -> LatentDisplacementCandidate:
    """Select the candidate with the best mean validation ROC AUC."""
    latent = validation_results.loc[validation_results["method"].eq("latent_displacement")]
    if latent.empty:
        raise ValueError("No latent-displacement validation results were provided")
    ranked = (
        latent.groupby(
            ["candidate_id", "mimic_capacity", "n_neighbors", "lambda_low", "lambda_high", "mimic_mode"],
            as_index=False,
        )["roc_auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["mean", "std", "candidate_id"], ascending=[False, True, True], na_position="last")
    )
    best = ranked.iloc[0]
    return LatentDisplacementCandidate(
        mimic_capacity=float(best["mimic_capacity"]),
        n_neighbors=int(best["n_neighbors"]),
        lambda_range=(float(best["lambda_low"]), float(best["lambda_high"])),
        mimic_mode=str(best["mimic_mode"]),
    )


def summarize_scores(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize ROC AUC and paired improvement over direct displacement."""
    if results.empty:
        return pd.DataFrame()
    keys = ["seed", "imbalance_ratio", "training_size"]
    direct = results.loc[results["method"].eq("direct_displacement"), keys + ["roc_auc"]].rename(
        columns={"roc_auc": "direct_roc_auc"}
    )
    scored = results.merge(direct, on=keys, how="left")
    scored["roc_auc_delta_vs_direct"] = scored["roc_auc"] - scored["direct_roc_auc"]
    group_keys = ["method", "candidate_id"]
    return (
        scored.groupby(group_keys, dropna=False, as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_delta_vs_direct=("roc_auc_delta_vs_direct", "mean"),
            wins_vs_direct=("roc_auc_delta_vs_direct", lambda values: int((values > 0).sum())),
            comparisons=("roc_auc", "size"),
        )
        .sort_values(["mean_roc_auc", "method"], ascending=[False, True])
        .reset_index(drop=True)
    )


def run_latent_displacement_tuning(
    config: ExperimentConfig,
    candidates: tuple[LatentDisplacementCandidate, ...],
    *,
    dataset_key: str = "default_credit",
    validation_size: float = 0.25,
    show_progress: bool = True,
) -> LatentDisplacementTuningResult:
    """Tune on inner validation splits, then compare once on outer held-out splits."""
    if not candidates:
        raise ValueError("At least one candidate is required")
    n_rows = config.profile.dataset_n_rows.get(dataset_key)
    frame = load_dataset(dataset_key, n_rows=n_rows, random_state=0)
    validation_rows: list[dict] = []
    outer_splits: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    total = len(config.profile.seeds) * len(config.profile.imbalance_ratios) * len(config.profile.training_sizes)
    total *= len(candidates) + 1
    completed = 0
    for seed in config.profile.seeds:
        outer_train, outer_test = _split(frame, test_size=config.test_size, seed=seed)
        outer_splits[seed] = (outer_train, outer_test)
        development, validation = _split(outer_train, test_size=validation_size, seed=seed + 10_000)
        prepared = fit_preprocess_train_test(development, validation, dataset_key=dataset_key, target_column=config.target_column)
        for ratio in config.profile.imbalance_ratios:
            for training_size in config.profile.training_sizes:
                imbalanced = make_imbalanced_subset(development, ratio=ratio, training_size=training_size, random_state=seed)
                validation_rows.append(_evaluate(config, None, dataset_key, imbalanced, prepared, seed, ratio, training_size))
                completed += 1
                _progress(completed, total, show_progress, "validation: direct")
                for candidate in candidates:
                    validation_rows.append(_evaluate(config, candidate, dataset_key, imbalanced, prepared, seed, ratio, training_size))
                    completed += 1
                    _progress(completed, total, show_progress, f"validation: {candidate.candidate_id}")

    validation_results = pd.DataFrame(validation_rows)
    selected = select_candidate(validation_results)
    heldout_rows: list[dict] = []
    heldout_total = len(config.profile.seeds) * len(config.profile.imbalance_ratios) * len(config.profile.training_sizes) * 2
    completed = 0
    for seed, (outer_train, outer_test) in outer_splits.items():
        prepared = fit_preprocess_train_test(outer_train, outer_test, dataset_key=dataset_key, target_column=config.target_column)
        for ratio in config.profile.imbalance_ratios:
            for training_size in config.profile.training_sizes:
                imbalanced = make_imbalanced_subset(outer_train, ratio=ratio, training_size=training_size, random_state=seed)
                for candidate in (None, selected):
                    heldout_rows.append(_evaluate(config, candidate, dataset_key, imbalanced, prepared, seed, ratio, training_size))
                    completed += 1
                    _progress(completed, heldout_total, show_progress, "held-out comparison")
    heldout_results = pd.DataFrame(heldout_rows)
    return LatentDisplacementTuningResult(
        validation_results=validation_results,
        validation_summary=summarize_scores(validation_results),
        selected_candidate=selected,
        heldout_results=heldout_results,
        heldout_summary=summarize_scores(heldout_results),
    )


def run_bayesian_latent_displacement_optimization(
    config: ExperimentConfig,
    *,
    dataset_key: str = "default_credit",
    space: BayesianOptimizationSpace = BayesianOptimizationSpace(),
    n_trials: int = 24,
    n_initial_points: int = 6,
    acquisition_pool_size: int = 2048,
    validation_size: float = 0.25,
    random_state: int = 0,
    show_progress: bool = True,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    restart: bool = False,
) -> LatentDisplacementTuningResult:
    """Maximize paired validation ROC-AUC advantage using GP expected improvement."""
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    n_initial_points = min(max(1, n_initial_points), n_trials)
    frame = load_dataset(
        dataset_key,
        n_rows=config.profile.dataset_n_rows.get(dataset_key),
        random_state=0,
    )
    contexts, outer_splits = _prepare_validation_contexts(
        config, frame, dataset_key=dataset_key, validation_size=validation_size
    )
    direct_rows = [
        _evaluate(config, None, dataset_key, imbalanced, prepared, seed, ratio, training_size)
        for seed, ratio, training_size, imbalanced, prepared in contexts
    ]
    validation_rows = list(direct_rows)
    direct_scores = np.asarray([row["roc_auc"] for row in direct_rows])
    direct_mean = float(np.mean(direct_scores))
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if restart and checkpoint is not None and checkpoint.exists():
        checkpoint.unlink()
    if restart and checkpoint is not None:
        _clear_persisted_tuning_results(checkpoint.parent)
    signature = _optimization_signature(config, dataset_key, space, validation_size, random_state)
    objectives: list[float] = []
    history_rows: list[dict] = []
    optimizer = Optimizer(
        dimensions=[
            Real(*space.capacity_bounds, name="mimic_capacity"),
            Integer(*space.neighbor_bounds, name="n_neighbors"),
            Real(*space.lambda_low_bounds, name="lambda_low"),
            Real(*space.lambda_width_bounds, name="lambda_width"),
        ],
        base_estimator="GP",
        acq_func="EI",
        acq_optimizer="sampling",
        acq_optimizer_kwargs={"n_points": acquisition_pool_size},
        n_initial_points=n_initial_points,
        random_state=random_state,
    )

    start_trial = 0
    if resume and checkpoint is not None and checkpoint.exists():
        state = joblib.load(checkpoint)
        if state.get("signature") != signature:
            raise ValueError(
                "The BO checkpoint does not match the current dataset, profile, or search space. "
                "Set RESTART_OPTIMIZATION = True to start a new run."
            )
        optimizer = state["optimizer"]
        objectives = state["objectives"]
        history_rows = state["history_rows"]
        validation_rows = state["validation_rows"]
        start_trial = len(objectives)
        if start_trial > n_trials:
            raise ValueError(
                f"Checkpoint already contains {start_trial} trials, more than requested n_trials={n_trials}. "
                "Increase N_TRIALS or restart the optimization."
            )
        if show_progress:
            print(f"Resuming Bayesian optimization at trial {start_trial + 1}/{n_trials} from {checkpoint}")

    if start_trial == 0 and checkpoint is not None:
        _clear_persisted_tuning_results(checkpoint.parent)

    for trial in range(start_trial, n_trials):
        point = optimizer.ask()
        candidate = _candidate_from_skopt_point(point, space)
        latent_rows = [
            _evaluate(config, candidate, dataset_key, imbalanced, prepared, seed, ratio, training_size)
            for seed, ratio, training_size, imbalanced, prepared in contexts
        ]
        validation_rows.extend(latent_rows)
        latent_scores = np.asarray([row["roc_auc"] for row in latent_rows])
        deltas = latent_scores - direct_scores
        objective = float(np.mean(deltas))
        optimizer.tell(point, -objective)
        objectives.append(objective)
        latent_mean = float(np.mean(latent_scores))
        best_trial_so_far = int(np.argmax(objectives))
        best_objective_so_far = float(objectives[best_trial_so_far])
        history_rows.append(
            {
                "trial": trial,
                "candidate_id": candidate.candidate_id,
                "mean_direct_roc_auc": direct_mean,
                "mean_latent_roc_auc": latent_mean,
                "objective_mean_delta_vs_direct": objective,
                "best_delta_so_far": best_objective_so_far,
                "best_trial_so_far": best_trial_so_far,
                "delta_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan"),
                "wins_vs_direct": int(np.sum(deltas > 0)),
                "comparisons": len(deltas),
                "optimizer_phase": "initial" if trial < n_initial_points else "bayesian",
                **_candidate_fields(candidate),
            }
        )
        if checkpoint is not None:
            _save_optimization_checkpoint(
                checkpoint,
                signature=signature,
                optimizer=optimizer,
                objectives=objectives,
                history_rows=history_rows,
                validation_rows=validation_rows,
            )
            _persist_partial_tuning_results(
                checkpoint.parent,
                validation_rows=validation_rows,
                history_rows=history_rows,
            )
        _progress(
            trial + 1,
            n_trials,
            show_progress,
            (
                f"direct={direct_mean:.5f} | latent={latent_mean:.5f} | "
                f"delta={objective:+.5f} | best={best_objective_so_far:+.5f} "
                f"(trial {best_trial_so_far + 1})"
            ),
        )

    best_trial = int(np.argmax(objectives))
    selected = _candidate_from_skopt_point(optimizer.Xi[best_trial], space)
    heldout_results = _evaluate_heldout(config, outer_splits, selected, dataset_key, show_progress)
    validation_results = pd.DataFrame(validation_rows)
    result = LatentDisplacementTuningResult(
        validation_results=validation_results,
        validation_summary=summarize_scores(validation_results),
        selected_candidate=selected,
        heldout_results=heldout_results,
        heldout_summary=summarize_scores(heldout_results),
        optimization_history=pd.DataFrame(history_rows).sort_values(
            "objective_mean_delta_vs_direct", ascending=False
        ).reset_index(drop=True),
    )
    if checkpoint is not None:
        save_tuning_result(result, checkpoint.parent.parent.parent)
    return result


def default_optimization_checkpoint_path(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / "tuning" / "latent_displacement_default_credit" / "bo_checkpoint.joblib"


def load_tuning_result(artifact_dir: str | Path) -> LatentDisplacementTuningResult:
    """Load the latest complete or partial tuning result from disk."""
    root = Path(artifact_dir) / "tuning" / "latent_displacement_default_credit"
    history_path = root / "optimization_history.csv"
    checkpoint_path = root / "bo_checkpoint.joblib"
    if checkpoint_path.exists() and (
        not history_path.exists() or checkpoint_path.stat().st_mtime_ns > history_path.stat().st_mtime_ns
    ):
        state = joblib.load(checkpoint_path)
        if state.get("history_rows"):
            _persist_partial_tuning_results(
                root,
                validation_rows=state["validation_rows"],
                history_rows=state["history_rows"],
            )
    history = _read_required_csv(history_path)
    validation = _read_required_csv(root / "validation_results.csv")
    heldout_path = root / "heldout_results.csv"
    heldout = pd.read_csv(heldout_path) if heldout_path.exists() else pd.DataFrame()
    return LatentDisplacementTuningResult(
        validation_results=validation,
        validation_summary=summarize_scores(validation),
        selected_candidate=_candidate_from_history(history),
        heldout_results=heldout,
        heldout_summary=summarize_scores(heldout),
        optimization_history=history,
    )




def save_tuning_result(result: LatentDisplacementTuningResult, artifact_dir: str | Path) -> pd.DataFrame:
    """Persist tuning tables and return a compact artifact manifest."""
    root = Path(artifact_dir) / "tuning" / "latent_displacement_default_credit"
    root.mkdir(parents=True, exist_ok=True)
    tables = {
        "validation_results": result.validation_results,
        "validation_summary": result.validation_summary,
        "heldout_results": result.heldout_results,
        "heldout_summary": result.heldout_summary,
    }
    if result.optimization_history is not None:
        tables["optimization_history"] = result.optimization_history
    rows = []
    for name, table in tables.items():
        path = root / f"{name}.csv"
        table.to_csv(path, index=False)
        rows.append({"artifact": name, "path": str(path), "rows": len(table)})
    selected_path = root / "selected_candidate.txt"
    selected_path.write_text(result.selected_candidate.candidate_id + "\n")
    rows.append({"artifact": "selected_candidate", "path": str(selected_path), "rows": 1})
    return pd.DataFrame(rows)


def _evaluate(config, candidate, dataset_key, imbalanced, prepared, seed, ratio, training_size) -> dict:
    method = "direct_displacement" if candidate is None else "latent_displacement"
    tuned = config if candidate is None else replace(
        config,
        mimic_capacity=candidate.mimic_capacity,
        n_neighbors=candidate.n_neighbors,
        lambda_range=candidate.lambda_range,
        mimic_mode=candidate.mimic_mode,
    )
    row = run_condition(
        tuned,
        dataset_key=dataset_key,
        imbalanced_raw=imbalanced,
        prepared=prepared,
        method=method,
        seed=seed,
        ratio=ratio,
        training_size=training_size,
        metadata={},
    )
    row.update(_candidate_fields(candidate))
    return row


def _candidate_from_skopt_point(point, space: BayesianOptimizationSpace) -> LatentDisplacementCandidate:
    capacity, neighbors, low, width = point
    return LatentDisplacementCandidate(
        mimic_capacity=float(capacity),
        n_neighbors=int(neighbors),
        lambda_range=(float(low), float(low + width)),
        mimic_mode=space.mimic_mode,
    )


def _prepare_validation_contexts(config, frame, *, dataset_key, validation_size):
    contexts = []
    outer_splits = {}
    for seed in config.profile.seeds:
        outer_train, outer_test = _split(frame, test_size=config.test_size, seed=seed)
        outer_splits[seed] = (outer_train, outer_test)
        development, validation = _split(outer_train, test_size=validation_size, seed=seed + 10_000)
        prepared = fit_preprocess_train_test(
            development, validation, dataset_key=dataset_key, target_column=config.target_column
        )
        for ratio in config.profile.imbalance_ratios:
            for training_size in config.profile.training_sizes:
                imbalanced = make_imbalanced_subset(
                    development, ratio=ratio, training_size=training_size, random_state=seed
                )
                contexts.append((seed, ratio, training_size, imbalanced, prepared))
    return contexts, outer_splits


def _evaluate_heldout(config, outer_splits, selected, dataset_key, show_progress):
    rows = []
    total = len(outer_splits) * len(config.profile.imbalance_ratios) * len(config.profile.training_sizes) * 2
    completed = 0
    for seed, (outer_train, outer_test) in outer_splits.items():
        prepared = fit_preprocess_train_test(
            outer_train, outer_test, dataset_key=dataset_key, target_column=config.target_column
        )
        for ratio in config.profile.imbalance_ratios:
            for training_size in config.profile.training_sizes:
                imbalanced = make_imbalanced_subset(
                    outer_train, ratio=ratio, training_size=training_size, random_state=seed
                )
                for candidate in (None, selected):
                    rows.append(
                        _evaluate(
                            config, candidate, dataset_key, imbalanced, prepared, seed, ratio, training_size
                        )
                    )
                    completed += 1
                    _progress(completed, total, show_progress, "held-out comparison")
    return pd.DataFrame(rows)


def _candidate_fields(candidate) -> dict:
    if candidate is None:
        return {"candidate_id": "direct_displacement", "mimic_capacity": None, "n_neighbors": None, "lambda_low": None, "lambda_high": None, "mimic_mode": None}
    return {
        "candidate_id": candidate.candidate_id,
        "mimic_capacity": candidate.mimic_capacity,
        "n_neighbors": candidate.n_neighbors,
        "lambda_low": candidate.lambda_range[0],
        "lambda_high": candidate.lambda_range[1],
        "mimic_mode": candidate.mimic_mode,
    }


def _optimization_signature(config, dataset_key, space, validation_size, random_state) -> str:
    payload = {
        "dataset_key": dataset_key,
        "seeds": list(config.profile.seeds),
        "imbalance_ratios": list(config.profile.imbalance_ratios),
        "training_sizes": list(config.profile.training_sizes),
        "test_size": config.test_size,
        "validation_size": validation_size,
        "space": {
            "capacity_bounds": list(space.capacity_bounds),
            "neighbor_bounds": list(space.neighbor_bounds),
            "lambda_low_bounds": list(space.lambda_low_bounds),
            "lambda_width_bounds": list(space.lambda_width_bounds),
            "mimic_mode": space.mimic_mode,
        },
        "random_state": random_state,
    }
    return json.dumps(payload, sort_keys=True)


def _save_optimization_checkpoint(path, *, signature, optimizer, objectives, history_rows, validation_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(
        {
            "signature": signature,
            "optimizer": optimizer,
            "objectives": objectives,
            "history_rows": history_rows,
            "validation_rows": validation_rows,
        },
        temporary,
    )
    temporary.replace(path)


def _persist_partial_tuning_results(root: Path, *, validation_rows, history_rows) -> None:
    root.mkdir(parents=True, exist_ok=True)
    validation = pd.DataFrame(validation_rows)
    history = pd.DataFrame(history_rows).sort_values(
        "objective_mean_delta_vs_direct", ascending=False
    ).reset_index(drop=True)
    for frame, path in (
        (validation, root / "validation_results.csv"),
        (summarize_scores(validation), root / "validation_summary.csv"),
        (history, root / "optimization_history.csv"),
    ):
        temporary_csv = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary_csv, index=False)
        temporary_csv.replace(path)
    best = history.loc[history["objective_mean_delta_vs_direct"].idxmax()]
    selected = LatentDisplacementCandidate(
        mimic_capacity=float(best["mimic_capacity"]),
        n_neighbors=int(best["n_neighbors"]),
        lambda_range=(float(best["lambda_low"]), float(best["lambda_high"])),
        mimic_mode=str(best["mimic_mode"]),
    )
    temporary = root / "selected_candidate.txt.tmp"
    temporary.write_text(selected.candidate_id + "\n")
    temporary.replace(root / "selected_candidate.txt")


def _candidate_from_history(history: pd.DataFrame) -> LatentDisplacementCandidate:
    if history.empty:
        raise ValueError("Optimization history is empty")
    best = history.loc[history["objective_mean_delta_vs_direct"].idxmax()]
    return LatentDisplacementCandidate(
        mimic_capacity=float(best["mimic_capacity"]),
        n_neighbors=int(best["n_neighbors"]),
        lambda_range=(float(best["lambda_low"]), float(best["lambda_high"])),
        mimic_mode=str(best["mimic_mode"]),
    )


def _clear_persisted_tuning_results(root: Path) -> None:
    for name in (
        "validation_results.csv",
        "validation_summary.csv",
        "optimization_history.csv",
        "heldout_results.csv",
        "heldout_summary.csv",
        "selected_candidate.txt",
    ):
        path = root / name
        if path.exists():
            path.unlink()


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No persisted optimization results found at {path}. Start the optimization first."
        )
    return pd.read_csv(path)


def _split(frame: pd.DataFrame, *, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(frame, test_size=test_size, random_state=seed, stratify=frame["label"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _progress(completed: int, total: int, enabled: bool, detail: str) -> None:
    if enabled:
        print(f"\r[{completed:>4}/{total}] {detail:<80}", end="\n" if completed == total else "", flush=True)
