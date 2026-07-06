"""Question 1 SMOTE-style ROC benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
import os
from pathlib import Path
import time

from joblib import Parallel, delayed, effective_n_jobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mimic import GenerationPolicy, MIMIC


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def print_progress_event(event: dict) -> None:
    completed = event.get("completed")
    total = event.get("total")
    fold = event.get("fold")
    parallel_workers = event.get("parallel_workers", 1)
    elapsed = event.get("elapsed")
    eta = event.get("eta")
    if completed == 0:
        print(f"Starting Q1 fold jobs: 0/{total} complete ({parallel_workers} worker(s))")
        return
    fold_text = f" fold {fold}" if fold is not None else ""
    eta_text = f", ETA {eta}" if eta is not None else ""
    print(f"Completed{fold_text}: {completed}/{total} folds, elapsed {elapsed}{eta_text}")


@dataclass(frozen=True)
class Q1Config:
    dataset_key: str = "adult_mixed"
    run_profile: str = "run_full"
    random_state: int = 0
    protocol: str = "deficit_sweep"
    mimic_mode: str = "factorised"
    mimic_capacity_run_full: float = 0.25
    deficit_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    smote_percent: int = 100
    under_sampling_percentages: tuple[int, ...] = (50, 100, 200, 300, 400, 500)
    n_jobs: int = 1
    artifact_dir: str | None = None
    cache_models: bool = True
    worker_threads: int = 1
    mimic_feature_n_jobs: int | None = 1
    policy: GenerationPolicy = field(
        default_factory=lambda: GenerationPolicy(
            method="smote",
            neighbour_mode="normal",
            n_neighbors=5,
            lambda_range=(0.0, 1.0),
        )
    )

    @property
    def n_splits(self) -> int:
        return 10

    @property
    def n_rows(self) -> int | None:
        return None

    @property
    def mimic_capacity(self) -> float:
        return self.mimic_capacity_run_full

    @property
    def saves_as_profile(self) -> str:
        return "run_full" if self.run_profile == "view" else self.run_profile

    @property
    def should_run_experiment(self) -> bool:
        return self.run_profile != "view"

    @property
    def is_paper_protocol(self) -> bool:
        return self.protocol == "paper_under_sampling"


def q1_dataset_registry(
    *,
    artifact_dir: str | Path | None = None,
    run_profile: str = "run_full",
    protocol: str = "deficit_sweep",
    mimic_mode: str = "factorised",
    mimic_capacity: float = 0.25,
    smote_percent: int = 100,
    under_sampling_percentages: tuple[int, ...] = (50, 100, 200, 300, 400, 500),
    policy: GenerationPolicy | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    registry = pd.DataFrame(
        [
            {"key": "pima", "dataset": "Pima", "majority": 500, "minority": 268, "status": "ready: OpenML data_id=37"},
            {
                "key": "phoneme",
                "dataset": "Phoneme",
                "majority": 3818,
                "minority": 1586,
                "status": "ready: OpenML data_id=1489",
            },
            {
                "key": "adult_mixed",
                "dataset": "Adult",
                "majority": 37155,
                "minority": 11687,
                "status": "ready: OpenML adult v2, mixed features",
            },
            {
                "key": "adult_numeric",
                "dataset": "Adult numeric-only",
                "majority": 37155,
                "minority": 11687,
                "status": "ready: OpenML adult v2, numeric features only",
            },
            {"key": "estate", "dataset": "E-state", "majority": 46869, "minority": 6351, "status": "optional: source needed"},
            {
                "key": "satimage",
                "dataset": "Satimage",
                "majority": 5809,
                "minority": 626,
                "status": "ready: OpenML data_id=182; smallest class vs rest",
            },
            {
                "key": "forest_cover",
                "dataset": "Forest Cover",
                "majority": 35754,
                "minority": 2747,
                "status": "ready: sklearn covtype classes 3 vs 4",
            },
            {"key": "oil", "dataset": "Oil", "majority": 896, "minority": 41, "status": "optional: source needed"},
            {
                "key": "mammography",
                "dataset": "Mammography",
                "majority": 10923,
                "minority": 260,
                "status": "ready: imbalanced-learn if installed; OpenML fallback",
            },
            {"key": "can", "dataset": "Can", "majority": 435512, "minority": 8360, "status": "optional: source needed"},
        ]
    )
    registry["experiment"] = [
        _experiment_status(
            key,
            artifact_dir=artifact_dir,
            run_profile=run_profile,
            protocol=protocol,
            mimic_mode=mimic_mode,
            mimic_capacity=mimic_capacity,
            smote_percent=smote_percent,
            under_sampling_percentages=under_sampling_percentages,
            policy=policy,
            random_state=random_state,
        )
        for key in registry["key"]
    ]
    return registry


def standardize_binary_frame(X, y, *, minority_value=None, target: str = "label") -> pd.DataFrame:
    frame = pd.DataFrame(X).copy()
    frame.columns = [_clean_column_name(column, i) for i, column in enumerate(frame.columns)]
    labels = pd.Series(y).reset_index(drop=True).astype("object")
    frame = frame.reset_index(drop=True)

    if minority_value is None:
        counts = labels.value_counts()
        if len(counts) != 2:
            raise ValueError(f"Expected a binary target, found {len(counts)} classes")
        minority_value = counts.idxmin()

    frame[target] = np.where(labels.eq(minority_value), "minority", "majority")
    return frame


def maybe_subsample(frame: pd.DataFrame, *, n_rows: int | None, random_state: int) -> pd.DataFrame:
    if n_rows is not None and n_rows < len(frame):
        return frame.sample(n=n_rows, random_state=random_state).reset_index(drop=True)
    return frame.reset_index(drop=True)


def load_q1_dataset(config: Q1Config) -> pd.DataFrame:
    key = config.dataset_key
    if key == "pima":
        return _load_openml_binary(data_id=37, n_rows=config.n_rows, random_state=config.random_state)
    if key == "phoneme":
        return _load_openml_binary(data_id=1489, n_rows=config.n_rows, random_state=config.random_state)
    if key == "adult_mixed":
        return _load_adult(numeric_only=False, n_rows=config.n_rows, random_state=config.random_state)
    if key == "adult_numeric":
        return _load_adult(numeric_only=True, n_rows=config.n_rows, random_state=config.random_state)
    if key == "satimage":
        return _load_satimage(n_rows=config.n_rows, random_state=config.random_state)
    if key == "forest_cover":
        return _load_forest_cover(n_rows=config.n_rows, random_state=config.random_state)
    if key == "mammography":
        return _load_mammography(n_rows=config.n_rows, random_state=config.random_state)
    if key in {"estate", "oil", "can"}:
        raise NotImplementedError(f"{key!r} needs the original SMOTE-paper source before it can be replicated")
    raise NotImplementedError(f"No loader implemented yet for {key!r}")


def infer_mimic_columns(frame: pd.DataFrame, target: str = "label") -> dict[str, list[str]]:
    feature_frame = frame.drop(columns=[target])
    numeric = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in feature_frame.columns if column not in numeric]
    return {"regression": numeric, "classification": categorical + [target]}


def generated_count_for_fraction(train: pd.DataFrame, fraction: float, target: str = "label") -> int:
    counts = train[target].value_counts()
    deficit = int(counts.get("majority", 0) - counts.get("minority", 0))
    return max(0, int(round(deficit * fraction)))


def generated_count_for_smote_percent(train: pd.DataFrame, smote_percent: int, target: str = "label") -> int:
    minority_count = int(train[target].eq("minority").sum())
    return max(0, int(round(minority_count * smote_percent / 100)))


def majority_count_for_under_sampling(
    augmented_train: pd.DataFrame, under_sampling_percent: int, target: str = "label"
) -> int:
    if under_sampling_percent <= 0:
        raise ValueError("under_sampling_percent must be positive")
    minority_count = int(augmented_train[target].eq("minority").sum())
    majority_available = int(augmented_train[target].eq("majority").sum())
    requested = int(round(minority_count * 100 / under_sampling_percent))
    return max(1, min(majority_available, requested))


def apply_majority_under_sampling(
    augmented_train: pd.DataFrame,
    *,
    under_sampling_percent: int,
    random_state: int,
    target: str = "label",
) -> pd.DataFrame:
    minority = augmented_train.loc[augmented_train[target].eq("minority")]
    majority = augmented_train.loc[augmented_train[target].eq("majority")]
    n_majority = majority_count_for_under_sampling(
        augmented_train,
        under_sampling_percent=under_sampling_percent,
        target=target,
    )
    sampled_majority = majority.sample(n=n_majority, random_state=random_state, replace=False)
    return pd.concat([minority, sampled_majority], ignore_index=True).sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)


def fit_mimic_and_augment(
    train: pd.DataFrame,
    n_generated: int,
    *,
    config: Q1Config,
    random_state: int,
    model: MIMIC | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_generated == 0:
        return train.copy(), pd.DataFrame()

    if model is None:
        model = fit_mimic_model(train, config=config, random_state=random_state)
    synthetic, trace = model.sample(n_generated, condition={"label": "minority"}, return_trace=True)
    augmented = pd.concat([train, synthetic], ignore_index=True)
    return augmented, trace


def fit_mimic_model(train: pd.DataFrame, *, config: Q1Config, random_state: int) -> MIMIC:
    model = MIMIC(
        columns=infer_mimic_columns(train),
        mode=config.mimic_mode,
        capacity=config.mimic_capacity,
        policy=config.policy,
        random_state=random_state,
        feature_n_jobs=config.mimic_feature_n_jobs,
    )
    model.fit(train)
    return model


def load_or_fit_mimic_model(
    train: pd.DataFrame,
    *,
    config: Q1Config,
    fold: int,
    train_idx,
    random_state: int,
) -> tuple[MIMIC, str | None, bool]:
    path = model_cache_path(config, fold=fold, train_idx=train_idx)
    if path is not None and path.exists():
        return MIMIC.load(path), str(path), True

    model = fit_mimic_model(train, config=config, random_state=random_state)
    if path is not None and config.cache_models:
        model.save(path)
        return model, str(path), False
    return model, str(path) if path is not None else None, False


def make_classifier(train: pd.DataFrame, *, random_state: int, target: str = "label") -> Pipeline:
    X = train.drop(columns=[target])
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("classifier", DecisionTreeClassifier(random_state=random_state, min_samples_leaf=5)),
        ]
    )


def evaluate_sampling_point(
    frame: pd.DataFrame,
    *,
    deficit_fraction: float,
    config: Q1Config,
    target: str = "label",
) -> dict[str, float]:
    splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    X = frame.drop(columns=[target])
    y = frame[target]
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        train = frame.iloc[train_idx].reset_index(drop=True)
        test = frame.iloc[test_idx].reset_index(drop=True)
        n_generated = generated_count_for_fraction(train, deficit_fraction, target=target)
        augmented, trace = fit_mimic_and_augment(train, n_generated, config=config, random_state=config.random_state + fold)

        classifier = make_classifier(augmented, random_state=config.random_state, target=target)
        classifier.fit(augmented.drop(columns=[target]), augmented[target])
        positive_class_index = list(classifier.classes_).index("minority")
        scores = classifier.predict_proba(test.drop(columns=[target]))[:, positive_class_index]
        y_true = test[target].eq("minority").astype(int).to_numpy()
        y_pred = scores >= 0.5

        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fpr = fp / (fp + tn) if (fp + tn) else np.nan
        tpr = tp / (tp + fn) if (tp + fn) else np.nan

        fold_rows.append(
            {
                "fold": fold,
                "deficit_fraction": deficit_fraction,
                "n_train": len(train),
                "n_generated": n_generated,
                "trace_rows": len(trace),
                "fpr": fpr,
                "tpr": tpr,
                "roc_auc_score": roc_auc_score(y_true, scores),
            }
        )

    fold_result = pd.DataFrame(fold_rows)
    return {
        "deficit_fraction": deficit_fraction,
        "mean_fpr": fold_result["fpr"].mean(),
        "mean_tpr": fold_result["tpr"].mean(),
        "mean_fold_auc": fold_result["roc_auc_score"].mean(),
        "mean_generated": fold_result["n_generated"].mean(),
        "folds": len(fold_result),
    }


def evaluate_fold(
    frame: pd.DataFrame,
    *,
    train_idx,
    test_idx,
    fold: int,
    config: Q1Config,
    target: str = "label",
) -> pd.DataFrame:
    _configure_worker_threads(config.worker_threads)
    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    model = None
    model_path = None
    model_cache_hit = False
    if any(generated_count_for_fraction(train, fraction, target=target) > 0 for fraction in config.deficit_fractions):
        model, model_path, model_cache_hit = load_or_fit_mimic_model(
            train,
            config=config,
            fold=fold,
            train_idx=train_idx,
            random_state=config.random_state + fold,
        )

    rows = []
    for deficit_fraction in config.deficit_fractions:
        n_generated = generated_count_for_fraction(train, deficit_fraction, target=target)
        augmented, trace = fit_mimic_and_augment(
            train,
            n_generated,
            config=config,
            random_state=config.random_state + fold,
            model=model,
        )
        classifier = make_classifier(augmented, random_state=config.random_state, target=target)
        classifier.fit(augmented.drop(columns=[target]), augmented[target])
        positive_class_index = list(classifier.classes_).index("minority")
        scores = classifier.predict_proba(test.drop(columns=[target]))[:, positive_class_index]
        y_true = test[target].eq("minority").astype(int).to_numpy()
        y_pred = scores >= 0.5

        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        rows.append(
            {
                "fold": fold,
                "deficit_fraction": deficit_fraction,
                "n_train": len(train),
                "n_generated": n_generated,
                "trace_rows": len(trace),
                "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
                "tpr": tp / (tp + fn) if (tp + fn) else np.nan,
                "roc_auc_score": roc_auc_score(y_true, scores),
                "model_path": model_path,
                "model_cache_hit": bool(model_cache_hit),
            }
        )
    return pd.DataFrame(rows)


def evaluate_paper_fold(
    frame: pd.DataFrame,
    *,
    train_idx,
    test_idx,
    fold: int,
    config: Q1Config,
    target: str = "label",
) -> pd.DataFrame:
    _configure_worker_threads(config.worker_threads)
    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    n_generated = generated_count_for_smote_percent(train, config.smote_percent, target=target)
    model = None
    model_path = None
    model_cache_hit = False
    if n_generated > 0:
        model, model_path, model_cache_hit = load_or_fit_mimic_model(
            train,
            config=config,
            fold=fold,
            train_idx=train_idx,
            random_state=config.random_state + fold,
        )
    augmented, trace = fit_mimic_and_augment(
        train,
        n_generated,
        config=config,
        random_state=config.random_state + fold,
        model=model,
    )

    rows = []
    for under_sampling_percent in config.under_sampling_percentages:
        sampled_train = apply_majority_under_sampling(
            augmented,
            under_sampling_percent=under_sampling_percent,
            random_state=config.random_state + fold * 1000 + int(under_sampling_percent),
            target=target,
        )
        classifier = make_classifier(sampled_train, random_state=config.random_state, target=target)
        classifier.fit(sampled_train.drop(columns=[target]), sampled_train[target])
        positive_class_index = list(classifier.classes_).index("minority")
        scores = classifier.predict_proba(test.drop(columns=[target]))[:, positive_class_index]
        y_true = test[target].eq("minority").astype(int).to_numpy()
        y_pred = scores >= 0.5

        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        rows.append(
            {
                "fold": fold,
                "smote_percent": config.smote_percent,
                "under_sampling_percent": under_sampling_percent,
                "n_train": len(train),
                "n_generated": n_generated,
                "n_sampled_train": len(sampled_train),
                "n_sampled_majority": int(sampled_train[target].eq("majority").sum()),
                "n_sampled_minority": int(sampled_train[target].eq("minority").sum()),
                "trace_rows": len(trace),
                "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
                "tpr": tp / (tp + fn) if (tp + fn) else np.nan,
                "roc_auc_score": roc_auc_score(y_true, scores),
                "model_path": model_path,
                "model_cache_hit": bool(model_cache_hit),
            }
        )
    return pd.DataFrame(rows)


def fold_indices(frame: pd.DataFrame, config: Q1Config, target: str = "label") -> list[tuple[int, np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    X = frame.drop(columns=[target])
    y = frame[target]
    return [(fold, train_idx, test_idx) for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1)]


def run_mimic_roc_sweep(frame: pd.DataFrame, config: Q1Config, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.is_paper_protocol:
        return run_paper_under_sampling_sweep(frame, config, progress=progress)

    fold_results = run_q1_fold_jobs(frame, config, progress=progress)
    points = [_summarize_sampling_point(fold_results, fraction) for fraction in config.deficit_fractions]
    roc_points = pd.DataFrame(points).sort_values(["mean_fpr", "mean_tpr"]).reset_index(drop=True)
    curve = roc_curve_points(roc_points)
    summary = pd.DataFrame(
        [
            {
                "dataset_key": config.dataset_key,
                "run_profile": config.run_profile,
                "n_rows": len(frame),
                "n_splits": config.n_splits,
                "n_jobs": config.n_jobs,
                "mimic_mode": config.mimic_mode,
                "mimic_capacity": config.mimic_capacity,
                "artifact_dir": config.artifact_dir,
                "roc_curve_auc": auc(curve["mean_fpr"], curve["mean_tpr"]),
                "mean_fold_auc_at_full_balance": roc_points.loc[
                    roc_points["deficit_fraction"].eq(1.0), "mean_fold_auc"
                ].mean(),
            }
        ]
    )
    save_q1_result_tables(config, roc_points, summary)
    return roc_points, summary


def run_paper_under_sampling_sweep(
    frame: pd.DataFrame, config: Q1Config, progress=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_results = run_q1_fold_jobs(frame, config, progress=progress)
    points = [
        _summarize_paper_sampling_point(fold_results, under_sampling_percent)
        for under_sampling_percent in config.under_sampling_percentages
    ]
    roc_points = pd.DataFrame(points).sort_values(["mean_fpr", "mean_tpr"]).reset_index(drop=True)
    curve = roc_curve_points(roc_points)
    summary = pd.DataFrame(
        [
            {
                "dataset_key": config.dataset_key,
                "run_profile": config.run_profile,
                "protocol": config.protocol,
                "n_rows": len(frame),
                "n_splits": config.n_splits,
                "n_jobs": config.n_jobs,
                "mimic_mode": config.mimic_mode,
                "mimic_capacity": config.mimic_capacity,
                "smote_percent": config.smote_percent,
                "under_sampling_percentages": ",".join(map(str, config.under_sampling_percentages)),
                "artifact_dir": config.artifact_dir,
                "roc_curve_auc": auc(curve["mean_fpr"], curve["mean_tpr"]),
                "mean_fold_auc_at_balanced_under_sampling": roc_points.loc[
                    roc_points["under_sampling_percent"].eq(100), "mean_fold_auc"
                ].mean(),
            }
        ]
    )
    save_q1_result_tables(config, roc_points, summary)
    return roc_points, summary


def run_q1_fold_jobs(frame: pd.DataFrame, config: Q1Config, progress=None) -> pd.DataFrame:
    jobs = fold_indices(frame, config)
    total = len(jobs)
    parallel_workers = _effective_worker_count(config.n_jobs, total)
    progress_state = {"completed": 0, "started_at": time.monotonic()}
    _emit_progress(
        progress,
        completed=0,
        total=total,
        started_at=progress_state["started_at"],
        fold=None,
        parallel_workers=parallel_workers,
    )
    if config.n_jobs == 1:
        results = []
        for fold, train_idx, test_idx in jobs:
            evaluator = evaluate_paper_fold if config.is_paper_protocol else evaluate_fold
            result = evaluator(frame, train_idx=train_idx, test_idx=test_idx, fold=fold, config=config)
            results.append(result)
            progress_state["completed"] += 1
            _emit_progress(
                progress,
                completed=progress_state["completed"],
                total=total,
                started_at=progress_state["started_at"],
                fold=fold,
                parallel_workers=parallel_workers,
            )
    else:
        evaluator = evaluate_paper_fold if config.is_paper_protocol else evaluate_fold
        task_iter = (
            delayed(evaluator)(frame, train_idx=train_idx, test_idx=test_idx, fold=fold, config=config)
            for fold, train_idx, test_idx in jobs
        )
        try:
            result_iter = Parallel(n_jobs=config.n_jobs, return_as="generator_unordered")(task_iter)
            results = []
            for result in result_iter:
                results.append(result)
                progress_state["completed"] += 1
                fold = int(result["fold"].iloc[0]) if not result.empty else None
                _emit_progress(
                    progress,
                    completed=progress_state["completed"],
                    total=total,
                    started_at=progress_state["started_at"],
                    fold=fold,
                    parallel_workers=parallel_workers,
                )
        except TypeError:
            task_iter = (
                delayed(evaluator)(frame, train_idx=train_idx, test_idx=test_idx, fold=fold, config=config)
                for fold, train_idx, test_idx in jobs
            )
            results = Parallel(n_jobs=config.n_jobs)(task_iter)
            _emit_progress(
                progress,
                completed=total,
                total=total,
                started_at=progress_state["started_at"],
                fold=None,
                parallel_workers=parallel_workers,
            )
    return pd.concat(results, ignore_index=True)


def roc_curve_points(roc_points: pd.DataFrame) -> pd.DataFrame:
    label_column = "deficit_fraction" if "deficit_fraction" in roc_points.columns else "under_sampling_percent"
    return pd.concat(
        [
            pd.DataFrame([{"mean_fpr": 0.0, "mean_tpr": 0.0, label_column: np.nan}]),
            roc_points[["mean_fpr", "mean_tpr", label_column]],
            pd.DataFrame([{"mean_fpr": 1.0, "mean_tpr": 1.0, label_column: np.nan}]),
        ],
        ignore_index=True,
    ).sort_values(["mean_fpr", "mean_tpr"]).reset_index(drop=True)


def plot_q1_roc_sweep(
    roc_points: pd.DataFrame,
    config: Q1Config,
    *,
    figsize: tuple[float, float] = (5.5, 5.0),
):
    curve = roc_curve_points(roc_points)
    label_column = "deficit_fraction" if "deficit_fraction" in roc_points.columns else "under_sampling_percent"
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(curve["mean_fpr"], curve["mean_tpr"], marker="o", linewidth=2, label="MIMIC ROC sweep")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.55)
    for _, row in roc_points.iterrows():
        label = f"{row[label_column]:.2g}" if label_column == "deficit_fraction" else f"{int(row[label_column])}%"
        ax.annotate(
            label,
            (row["mean_fpr"], row["mean_tpr"]),
            textcoords="offset points",
            xytext=(5, 5),
        )
    ax.set_title(f"Q1 MIMIC ROC sweep: {config.dataset_key}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def manuscript_result_row(config: Q1Config, summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": config.dataset_key,
                "classifier": "sklearn DecisionTreeClassifier",
                "published_smote_reference": "SMOTE paper ROC/AUC target",
                "mimic_auc": summary.loc[0, "roc_curve_auc"],
                "mimic_roc_hull_status": "pending hull comparison",
                "protocol_note": (
                    "paper under-sampling sweep, classifier-substituted"
                    if config.is_paper_protocol
                    else "protocol-aligned, classifier-substituted"
                ),
            }
        ]
    )


def save_q1_result_tables(config: Q1Config, roc_points: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Path] | None:
    paths = q1_result_table_paths(config)
    if paths is None:
        return None
    paths["roc_points"].parent.mkdir(parents=True, exist_ok=True)
    roc_points.to_csv(paths["roc_points"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return paths


def load_q1_result_tables(config: Q1Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = q1_result_table_paths(config)
    if paths is None:
        raise ValueError("Q1 result tables require config.artifact_dir")
    return pd.read_csv(paths["roc_points"]), pd.read_csv(paths["summary"])


def q1_result_table_manifest(config: Q1Config) -> pd.DataFrame:
    paths = q1_result_table_paths(config)
    if paths is None:
        return pd.DataFrame([{"table": "roc_points", "path": None}, {"table": "summary", "path": None}])
    return pd.DataFrame(
        [
            {"table": name, "path": str(path)}
            for name, path in paths.items()
        ]
    )


def q1_result_table_paths(config: Q1Config) -> dict[str, Path] | None:
    if config.artifact_dir is None:
        return None
    base = Path(config.artifact_dir) / "tables" / _config_artifact_stem(config)
    return {
        "roc_points": base.with_name(f"{base.name}__roc_points.csv"),
        "summary": base.with_name(f"{base.name}__summary.csv"),
    }


def model_cache_path(config: Q1Config, *, fold: int, train_idx) -> Path | None:
    if config.artifact_dir is None or not config.cache_models:
        return None
    train_hash = _index_hash(train_idx)
    name = (
        f"{_config_artifact_stem(config)}__fold-{fold}__train-{train_hash}.joblib"
    )
    return Path(config.artifact_dir) / "models" / name


def _experiment_status(
    dataset_key: str,
    *,
    artifact_dir: str | Path | None,
    run_profile: str,
    protocol: str,
    mimic_mode: str,
    mimic_capacity: float,
    smote_percent: int,
    under_sampling_percentages: tuple[int, ...],
    policy: GenerationPolicy | None,
    random_state: int,
) -> str:
    if artifact_dir is None:
        return "unknown: no artifact_dir"

    selected_policy = policy or GenerationPolicy(
        method="smote",
        neighbour_mode="normal",
        n_neighbors=5,
        lambda_range=(0.0, 1.0),
    )
    saved_profile = "run_full" if run_profile == "view" else run_profile
    expected_folds = 10
    protocol_part = "" if protocol == "deficit_sweep" else f"{_safe_name(protocol)}__"
    paper_part = ""
    if protocol == "paper_under_sampling":
        paper_part = f"smote-{smote_percent}__under-{'-'.join(map(str, under_sampling_percentages))}__"
    stem = (
        f"{_safe_name(dataset_key)}__{_safe_name(saved_profile)}__{protocol_part}"
        f"{_safe_name(mimic_mode)}__cap-{mimic_capacity:g}__{paper_part}"
        f"{_safe_name(selected_policy.method)}-{_safe_name(selected_policy.neighbour_mode)}-"
        f"k{selected_policy.n_neighbors}__seed-{random_state}"
    )
    summary_path = Path(artifact_dir) / "tables" / f"{stem}__summary.csv"
    if summary_path.exists():
        return "complete: current summary csv"

    existing_summaries = sorted((Path(artifact_dir) / "tables").glob(f"{_safe_name(dataset_key)}__run_full__*__summary.csv"))
    if existing_summaries:
        labels = [_summary_label(path, dataset_key=dataset_key) for path in existing_summaries[:2]]
        suffix = "" if len(existing_summaries) <= 2 else f"; +{len(existing_summaries) - 2} more"
        return f"existing: {'; '.join(labels)}{suffix}"

    prefix = f"{stem}__fold-"
    completed_folds = {
        int(path.name.removeprefix(prefix).split("__", 1)[0])
        for path in (Path(artifact_dir) / "models").glob(f"{prefix}*.joblib")
        if path.name.removeprefix(prefix).split("__", 1)[0].isdigit()
    }
    completed = len(completed_folds)
    if completed >= expected_folds:
        return f"complete: {expected_folds}/{expected_folds} folds"
    if completed:
        return f"partial: {completed}/{expected_folds} folds"
    return "not run"


def _emit_progress(
    progress,
    *,
    completed,
    total: int,
    started_at: float,
    fold: int | None,
    now: float | None = None,
    parallel_workers: int = 1,
) -> None:
    if progress is None:
        return
    current = time.monotonic() if now is None else now
    elapsed = current - started_at
    eta = None
    if completed is not None and completed > 0:
        eta = elapsed / completed * max(0, total - completed) / max(1, parallel_workers)
    event = {
        "completed": completed,
        "total": total,
        "fold": fold,
        "parallel_workers": parallel_workers,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "elapsed": format_seconds(elapsed),
        "eta": format_seconds(eta) if eta is not None else None,
    }
    progress(event)


def _summarize_sampling_point(fold_results: pd.DataFrame, deficit_fraction: float) -> dict[str, float]:
    selected = fold_results.loc[fold_results["deficit_fraction"].eq(deficit_fraction)]
    return {
        "deficit_fraction": deficit_fraction,
        "mean_fpr": selected["fpr"].mean(),
        "mean_tpr": selected["tpr"].mean(),
        "mean_fold_auc": selected["roc_auc_score"].mean(),
        "mean_generated": selected["n_generated"].mean(),
        "folds": len(selected),
        "model_cache_hits": int(selected["model_cache_hit"].sum()),
    }


def _summarize_paper_sampling_point(fold_results: pd.DataFrame, under_sampling_percent: int) -> dict[str, float]:
    selected = fold_results.loc[fold_results["under_sampling_percent"].eq(under_sampling_percent)]
    return {
        "smote_percent": int(selected["smote_percent"].iloc[0]) if not selected.empty else np.nan,
        "under_sampling_percent": under_sampling_percent,
        "mean_fpr": selected["fpr"].mean(),
        "mean_tpr": selected["tpr"].mean(),
        "mean_fold_auc": selected["roc_auc_score"].mean(),
        "mean_generated": selected["n_generated"].mean(),
        "mean_sampled_majority": selected["n_sampled_majority"].mean(),
        "mean_sampled_minority": selected["n_sampled_minority"].mean(),
        "folds": len(selected),
        "model_cache_hits": int(selected["model_cache_hit"].sum()),
    }


def _clean_column_name(column, index: int) -> str:
    name = str(column).strip().replace(" ", "_").replace("-", "_")
    return name if name else f"x{index}"


def _load_openml_binary(*, data_id=None, name=None, minority_value=None, n_rows: int | None, random_state: int) -> pd.DataFrame:
    dataset = fetch_openml(data_id=data_id, name=name, as_frame=True)
    frame = standardize_binary_frame(dataset.data, dataset.target, minority_value=minority_value)
    return maybe_subsample(frame, n_rows=n_rows, random_state=random_state)


def _index_hash(indices) -> str:
    arr = np.asarray(indices, dtype=np.int64)
    return blake2b(arr.tobytes(), digest_size=8).hexdigest()


def _config_artifact_stem(config: Q1Config) -> str:
    policy = config.policy
    protocol = "" if config.protocol == "deficit_sweep" else f"{_safe_name(config.protocol)}__"
    paper = ""
    if config.is_paper_protocol:
        paper = f"smote-{config.smote_percent}__under-{'-'.join(map(str, config.under_sampling_percentages))}__"
    return (
        f"{_safe_name(config.dataset_key)}__{_safe_name(config.saves_as_profile)}__{protocol}"
        f"{_safe_name(config.mimic_mode)}__cap-{config.mimic_capacity:g}__"
        f"{paper}"
        f"{_safe_name(policy.method)}-{_safe_name(policy.neighbour_mode)}-k{policy.n_neighbors}__"
        f"seed-{config.random_state}"
    )


def _safe_name(value) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in str(value))


def _summary_label(path: Path, *, dataset_key: str) -> str:
    stem = path.name.removesuffix("__summary.csv")
    prefix = f"{_safe_name(dataset_key)}__run_full__"
    if stem.startswith(prefix):
        stem = stem.removeprefix(prefix)
    return stem.replace("__", " ")


def _configure_worker_threads(worker_threads: int) -> None:
    value = str(max(1, int(worker_threads)))
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(name, value)
    try:
        import torch

        torch.set_num_threads(int(value))
    except Exception:
        pass


def _effective_worker_count(n_jobs: int, total: int) -> int:
    try:
        workers = effective_n_jobs(n_jobs)
    except Exception:
        workers = 1
    return max(1, min(int(workers), int(total)))


def _load_adult(*, numeric_only: bool, n_rows: int | None, random_state: int) -> pd.DataFrame:
    adult = fetch_openml("adult", version=2, as_frame=True)
    raw = adult.frame.copy().replace("?", np.nan)
    clean = raw.drop(columns=["fnlwgt", "education-num"]).dropna().reset_index(drop=True)
    clean["label"] = np.where(clean["class"].astype(str).str.strip().eq(">50K"), "minority", "majority")
    clean = clean.drop(columns=["class"])

    if numeric_only:
        numeric_columns = clean.select_dtypes(include=[np.number]).columns.tolist()
        clean = clean[numeric_columns + ["label"]]

    return maybe_subsample(clean, n_rows=n_rows, random_state=random_state)


def _load_satimage(*, n_rows: int | None, random_state: int) -> pd.DataFrame:
    dataset = fetch_openml(data_id=182, as_frame=True)
    labels = pd.Series(dataset.target).astype("object")
    minority_value = labels.value_counts().idxmin()
    frame = standardize_binary_frame(dataset.data, labels, minority_value=minority_value)
    return maybe_subsample(frame, n_rows=n_rows, random_state=random_state)


def _load_forest_cover(*, n_rows: int | None, random_state: int) -> pd.DataFrame:
    covtype = fetch_covtype(as_frame=True)
    X = covtype.data
    y = pd.Series(covtype.target)
    mask = y.isin([3, 4])
    frame = standardize_binary_frame(
        X.loc[mask].reset_index(drop=True),
        y.loc[mask].reset_index(drop=True),
        minority_value=4,
    )
    return maybe_subsample(frame, n_rows=n_rows, random_state=random_state)


def _load_mammography(*, n_rows: int | None, random_state: int) -> pd.DataFrame:
    try:
        from imblearn.datasets import fetch_datasets

        bunch = fetch_datasets()["mammography"]
        frame = standardize_binary_frame(bunch.data, bunch.target)
    except Exception:
        dataset = fetch_openml(name="mammography", as_frame=True)
        frame = standardize_binary_frame(dataset.data, dataset.target)
    return maybe_subsample(frame, n_rows=n_rows, random_state=random_state)
