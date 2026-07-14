"""Experiment orchestration for the streamlined framework."""

from __future__ import annotations

from pathlib import Path
from hashlib import blake2b
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from .analysis import (
    aulc_table,
    learning_curves,
    pairwise_comparisons,
    prescriptive_conclusions,
    rank_summary,
    real_equivalent_sample_fraction,
    real_equivalent_summary,
    regime_summary,
)
from .config import ExperimentConfig, artifact_paths, ensure_artifact_dirs, load_config
from .datasets import dataset_metadata, dataset_registry, load_dataset
from .metrics import empty_results, evaluate_binary_classifier, result_schema
from .plotting import save_all_figures, save_critical_difference_figures
from .preprocessing import fit_preprocess_train_test
from .sampling import build_balanced_training_set, make_imbalanced_subset


def run_profile(
    config_or_path: ExperimentConfig | str | Path,
    *,
    run_experiment: bool = True,
    show_progress: bool = False,
    restart: bool = False,
) -> dict[str, pd.DataFrame]:
    config = load_config(config_or_path) if not isinstance(config_or_path, ExperimentConfig) else config_or_path
    ensure_artifact_dirs(config)
    paths = artifact_paths(config)
    if run_experiment:
        if restart and paths["raw_results"].exists():
            paths["raw_results"].unlink()
        raw = run_conditions(config, show_progress=show_progress, raw_results_path=paths["raw_results"])
    else:
        raw = _read_csv(paths["raw_results"], empty_results())
    tables = build_analysis_artifacts(config, raw)
    return {"raw_results": raw, **tables}


def run_conditions(config: ExperimentConfig, *, show_progress: bool = False, raw_results_path: Path | None = None) -> pd.DataFrame:
    existing = _read_csv(raw_results_path, empty_results()) if raw_results_path is not None else empty_results()
    completed_keys = _completed_condition_keys(existing)
    rows = existing.to_dict("records") if not existing.empty else []
    total = _total_conditions(config)
    completed = len(completed_keys)
    started_at = time.monotonic()
    _emit_progress(completed, total, show_progress=show_progress, started_at=started_at)
    for dataset_key in config.profile.datasets:
        n_rows = config.profile.dataset_n_rows.get(dataset_key)
        frame = load_dataset(dataset_key, n_rows=n_rows, random_state=0)
        for seed in config.profile.seeds:
            train, test = train_test_split(
                frame,
                test_size=config.test_size,
                random_state=seed,
                stratify=frame["label"],
            )
            train = train.reset_index(drop=True)
            test = test.reset_index(drop=True)
            prepared = fit_preprocess_train_test(train, test, dataset_key=dataset_key, target_column=config.target_column)
            baseline_auc = _baseline_auc(prepared, seed)
            metadata = dataset_metadata(frame, dataset_key, preprocessed_dim=prepared.X_train.shape[1], baseline_roc_auc=baseline_auc)
            for ratio in config.profile.imbalance_ratios:
                for training_size in config.profile.training_sizes:
                    imbalanced = make_imbalanced_subset(
                        train,
                        ratio=ratio,
                        training_size=training_size,
                        random_state=_condition_seed(seed, ratio, training_size),
                    )
                    for method in config.profile.methods:
                        key = _condition_key(dataset_key, ratio, training_size, seed, method)
                        if key in completed_keys:
                            continue
                        completed += 1
                        _emit_progress(
                            completed,
                            total,
                            show_progress=show_progress,
                            dataset_key=dataset_key,
                            ratio=ratio,
                            training_size=training_size,
                            seed=seed,
                            method=method,
                            started_at=started_at,
                        )
                        row = run_condition(
                            config,
                            dataset_key=dataset_key,
                            imbalanced_raw=imbalanced,
                            train_pool_raw=train,
                            prepared=prepared,
                            method=method,
                            seed=seed,
                            ratio=ratio,
                            training_size=training_size,
                            metadata=metadata,
                        )
                        rows.append(row)
                        completed_keys.add(key)
                        if raw_results_path is not None:
                            _append_csv_row(row, raw_results_path, columns=result_schema() + _metadata_columns())
    return _dedupe_results(pd.DataFrame(rows, columns=result_schema() + _metadata_columns()))


def run_condition(
    config: ExperimentConfig,
    *,
    dataset_key: str,
    imbalanced_raw: pd.DataFrame,
    prepared,
    train_pool_raw: pd.DataFrame | None = None,
    method: str,
    seed: int,
    ratio: float,
    training_size: int,
    metadata: dict,
) -> dict:
    balanced = build_balanced_training_set(
        method,
        imbalanced_raw=imbalanced_raw,
        train_pool_raw=train_pool_raw,
        prepared=prepared,
        dataset_key=dataset_key,
        random_state=_condition_seed(seed, ratio, training_size, method),
        n_neighbors=config.n_neighbors,
        lambda_range=config.lambda_range,
        mimic_mode=config.mimic_mode,
        mimic_capacity=config.mimic_capacity,
        repair_direct_samples=config.repair_direct_samples,
    )
    classifier = HistGradientBoostingClassifier(random_state=seed)
    classifier.fit(balanced.X, balanced.y)
    y_score = classifier.predict_proba(prepared.X_test)[:, 1]
    y_pred = classifier.predict(prepared.X_test)
    metrics = evaluate_binary_classifier(prepared.y_test, y_score, y_pred)
    return {
        "dataset_key": dataset_key,
        "imbalance_ratio": ratio,
        "training_size": training_size,
        "seed": seed,
        "method": method,
        "generated_count": balanced.generated_count,
        "real_minority_count": balanced.real_minority_count,
        "real_majority_count": balanced.real_majority_count,
        **metrics,
        **metadata,
    }


def build_analysis_artifacts(config: ExperimentConfig, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    paths = artifact_paths(config)
    learning = learning_curves(raw)
    aulc = aulc_table(raw, config)
    pairwise = pairwise_comparisons(aulc, config) if not aulc.empty else pd.DataFrame()
    rank = rank_summary(aulc)
    real_equivalence = real_equivalent_sample_fraction(learning)
    real_equivalence_summary = real_equivalent_summary(real_equivalence)
    regime = regime_summary(aulc, pairwise)
    _write_csv(learning, paths["learning_curves"])
    _write_csv(aulc, paths["aulc"])
    _write_csv(pairwise, paths["pairwise"])
    _write_csv(real_equivalence, paths["real_equivalence"])
    _write_csv(real_equivalence_summary, paths["real_equivalence_summary"])
    _write_csv(regime, paths["regime"])
    _write_csv(rank, paths["rank"])
    paths["conclusions"].write_text(prescriptive_conclusions(regime))
    save_all_figures(learning, rank, paths["figures"])
    save_critical_difference_figures(aulc, paths["figures"])
    return {
        "learning_curves": learning,
        "aulc": aulc,
        "pairwise": pairwise,
        "real_equivalence": real_equivalence,
        "real_equivalence_summary": real_equivalence_summary,
        "regime": regime,
        "rank": rank,
        "registry": dataset_registry(),
    }


def artifact_manifest(config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for key, path in artifact_paths(config).items():
        try:
            relative_path = path.relative_to(config.artifact_root)
            base_dir = config.artifact_root
        except ValueError:
            relative_path = path.name
            base_dir = path.parent
        rows.append(
            {
                "artifact": key,
                "base_dir": str(base_dir),
                "relative_path": str(relative_path),
                "path": str(path),
                "exists": path.exists(),
            }
        )
    return pd.DataFrame(rows)


def _baseline_auc(prepared, seed: int) -> float:
    if len(np.unique(prepared.y_train)) < 2:
        return float("nan")
    classifier = HistGradientBoostingClassifier(random_state=seed)
    classifier.fit(prepared.X_train, prepared.y_train)
    scores = classifier.predict_proba(prepared.X_test)[:, 1]
    return evaluate_binary_classifier(prepared.y_test, scores)["roc_auc"]


def _condition_seed(seed: int, *parts) -> int:
    text = "|".join([str(seed), *(str(p) for p in parts)])
    digest = blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little", signed=False)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _append_csv_row(row: dict, path: Path, *, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{column: row.get(column) for column in columns}], columns=columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _read_csv(path: Path, default: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return default
    return pd.read_csv(path)


def _metadata_columns() -> list[str]:
    return [
        "n_samples",
        "n_features",
        "n_numeric_features",
        "n_categorical_features",
        "original_imbalance_ratio",
        "minority_size",
        "preprocessed_dim",
        "missingness_rate",
        "baseline_roc_auc",
    ]


def _condition_key(dataset_key: str, ratio: float, training_size: int, seed: int, method: str) -> tuple:
    return (str(dataset_key), float(ratio), int(training_size), int(seed), str(method))


def _completed_condition_keys(results: pd.DataFrame) -> set[tuple]:
    if results.empty:
        return set()
    required = {"dataset_key", "imbalance_ratio", "training_size", "seed", "method"}
    if not required.issubset(results.columns):
        return set()
    return {
        _condition_key(row.dataset_key, row.imbalance_ratio, row.training_size, row.seed, row.method)
        for row in results[list(required)].itertuples(index=False)
    }


def _dedupe_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    key_columns = ["dataset_key", "imbalance_ratio", "training_size", "seed", "method"]
    present = [column for column in key_columns if column in results.columns]
    if len(present) != len(key_columns):
        return results
    return results.drop_duplicates(subset=key_columns, keep="last").reset_index(drop=True)


def _total_conditions(config: ExperimentConfig) -> int:
    return (
        len(config.profile.datasets)
        * len(config.profile.seeds)
        * len(config.profile.imbalance_ratios)
        * len(config.profile.training_sizes)
        * len(config.profile.methods)
    )


def _emit_progress(
    completed: int,
    total: int,
    *,
    show_progress: bool,
    dataset_key: str | None = None,
    ratio: float | None = None,
    training_size: int | None = None,
    seed: int | None = None,
    method: str | None = None,
    started_at: float | None = None,
) -> None:
    if not show_progress:
        return
    try:
        from IPython.display import clear_output
    except Exception:
        clear_output = None
    if clear_output is not None:
        clear_output(wait=True)
    width = 30
    filled = int(round(width * completed / total)) if total else width
    bar = "#" * filled + "-" * (width - filled)
    detail = ""
    if dataset_key is not None:
        detail = f" {dataset_key} ratio={ratio:g}:1 n={training_size} seed={seed} method={method}"
    timing = _progress_timing(completed, total, started_at)
    print(f"[{bar}] {completed}/{total}{timing}{detail}", flush=True)


def _progress_timing(completed: int, total: int, started_at: float | None) -> str:
    if started_at is None:
        return ""
    elapsed = max(0.0, time.monotonic() - started_at)
    if completed <= 0:
        return f" elapsed={_format_duration(elapsed)} ETA=unknown"
    avg = elapsed / completed
    remaining = max(0, total - completed)
    eta = avg * remaining
    return f" elapsed={_format_duration(elapsed)} avg={_format_duration(avg)}/step ETA={_format_duration(eta)}"


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"
