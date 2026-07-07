"""Experiment orchestration for the streamlined framework."""

from __future__ import annotations

from pathlib import Path
from hashlib import blake2b

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from .analysis import aulc_table, learning_curves, pairwise_comparisons, prescriptive_conclusions, rank_summary, regime_summary
from .config import ExperimentConfig, artifact_paths, ensure_artifact_dirs, load_config
from .datasets import dataset_metadata, dataset_registry, load_dataset
from .metrics import empty_results, evaluate_binary_classifier, result_schema
from .plotting import save_all_figures
from .preprocessing import fit_preprocess_train_test
from .sampling import build_balanced_training_set, make_imbalanced_subset


def run_profile(config_or_path: ExperimentConfig | str | Path, *, run_experiment: bool = True) -> dict[str, pd.DataFrame]:
    config = load_config(config_or_path) if not isinstance(config_or_path, ExperimentConfig) else config_or_path
    ensure_artifact_dirs(config)
    paths = artifact_paths(config)
    if run_experiment:
        raw = run_conditions(config)
        _write_csv(raw, paths["raw_results"])
    else:
        raw = _read_csv(paths["raw_results"], empty_results())
    tables = build_analysis_artifacts(config, raw)
    return {"raw_results": raw, **tables}


def run_conditions(config: ExperimentConfig) -> pd.DataFrame:
    rows = []
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
                        rows.append(
                            run_condition(
                                config,
                                dataset_key=dataset_key,
                                imbalanced_raw=imbalanced,
                                prepared=prepared,
                                method=method,
                                seed=seed,
                                ratio=ratio,
                                training_size=training_size,
                                metadata=metadata,
                            )
                        )
    return pd.DataFrame(rows, columns=result_schema() + _metadata_columns())


def run_condition(
    config: ExperimentConfig,
    *,
    dataset_key: str,
    imbalanced_raw: pd.DataFrame,
    prepared,
    method: str,
    seed: int,
    ratio: float,
    training_size: int,
    metadata: dict,
) -> dict:
    balanced = build_balanced_training_set(
        method,
        imbalanced_raw=imbalanced_raw,
        prepared=prepared,
        dataset_key=dataset_key,
        random_state=_condition_seed(seed, ratio, training_size, method),
        n_neighbors=config.n_neighbors,
        lambda_range=config.lambda_range,
        mimic_mode=config.mimic_mode,
        mimic_capacity=config.mimic_capacity,
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
    regime = regime_summary(aulc, pairwise)
    _write_csv(learning, paths["learning_curves"])
    _write_csv(aulc, paths["aulc"])
    _write_csv(pairwise, paths["pairwise"])
    _write_csv(regime, paths["regime"])
    _write_csv(rank, paths["rank"])
    paths["conclusions"].write_text(prescriptive_conclusions(regime))
    save_all_figures(learning, rank, paths["figures"])
    return {"learning_curves": learning, "aulc": aulc, "pairwise": pairwise, "regime": regime, "rank": rank, "registry": dataset_registry()}


def artifact_manifest(config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for key, path in artifact_paths(config).items():
        rows.append({"artifact": key, "path": str(path), "exists": path.exists()})
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
