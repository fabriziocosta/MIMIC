"""Question 1 SMOTE-style ROC benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
import os
from pathlib import Path
import time

from joblib import Parallel, delayed
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
    elapsed = event.get("elapsed")
    eta = event.get("eta")
    if completed == 0:
        print(f"Starting Q1 fold jobs: 0/{total} complete")
        return
    fold_text = f" fold {fold}" if fold is not None else ""
    eta_text = f", ETA {eta}" if eta is not None else ""
    print(f"Completed{fold_text}: {completed}/{total} folds, elapsed {elapsed}{eta_text}")


@dataclass(frozen=True)
class Q1Config:
    dataset_key: str = "adult_mixed"
    run_profile: str = "smoke"
    random_state: int = 0
    n_rows_smoke: int = 250
    mimic_mode: str = "factorised"
    mimic_capacity_smoke: float = 0.0
    mimic_capacity_paper: float = 0.25
    deficit_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    n_jobs: int = 1
    artifact_dir: str | None = None
    cache_models: bool = True
    worker_threads: int = 1
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
        return 3 if self.run_profile == "smoke" else 10

    @property
    def n_rows(self) -> int | None:
        return self.n_rows_smoke if self.run_profile == "smoke" else None

    @property
    def mimic_capacity(self) -> float:
        return self.mimic_capacity_smoke if self.run_profile == "smoke" else self.mimic_capacity_paper


def q1_dataset_registry() -> pd.DataFrame:
    return pd.DataFrame(
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
                "status": "ready: OpenML name=satimage; smallest class vs rest",
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


def fold_indices(frame: pd.DataFrame, config: Q1Config, target: str = "label") -> list[tuple[int, np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    X = frame.drop(columns=[target])
    y = frame[target]
    return [(fold, train_idx, test_idx) for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1)]


def run_mimic_roc_sweep(frame: pd.DataFrame, config: Q1Config, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return roc_points, summary


def run_q1_fold_jobs(frame: pd.DataFrame, config: Q1Config, progress=None) -> pd.DataFrame:
    jobs = fold_indices(frame, config)
    total = len(jobs)
    progress_state = {"completed": 0, "started_at": time.monotonic()}
    _emit_progress(progress, completed=0, total=total, started_at=progress_state["started_at"], fold=None)
    if config.n_jobs == 1:
        results = []
        for fold, train_idx, test_idx in jobs:
            result = evaluate_fold(frame, train_idx=train_idx, test_idx=test_idx, fold=fold, config=config)
            results.append(result)
            progress_state["completed"] += 1
            _emit_progress(
                progress,
                completed=progress_state["completed"],
                total=total,
                started_at=progress_state["started_at"],
                fold=fold,
            )
    else:
        task_iter = (
            delayed(evaluate_fold)(frame, train_idx=train_idx, test_idx=test_idx, fold=fold, config=config)
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
                )
        except TypeError:
            task_iter = (
                delayed(evaluate_fold)(frame, train_idx=train_idx, test_idx=test_idx, fold=fold, config=config)
                for fold, train_idx, test_idx in jobs
            )
            results = Parallel(n_jobs=config.n_jobs)(task_iter)
            _emit_progress(progress, completed=total, total=total, started_at=progress_state["started_at"], fold=None)
    return pd.concat(results, ignore_index=True)


def roc_curve_points(roc_points: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame([{"mean_fpr": 0.0, "mean_tpr": 0.0, "deficit_fraction": np.nan}]),
            roc_points[["mean_fpr", "mean_tpr", "deficit_fraction"]],
            pd.DataFrame([{"mean_fpr": 1.0, "mean_tpr": 1.0, "deficit_fraction": np.nan}]),
        ],
        ignore_index=True,
    ).sort_values(["mean_fpr", "mean_tpr"]).reset_index(drop=True)


def manuscript_result_row(config: Q1Config, summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": config.dataset_key,
                "classifier": "sklearn DecisionTreeClassifier",
                "published_smote_reference": "SMOTE paper ROC/AUC target",
                "mimic_auc": summary.loc[0, "roc_curve_auc"],
                "mimic_roc_hull_status": "pending hull comparison",
                "protocol_note": "protocol-aligned, classifier-substituted",
            }
        ]
    )


def model_cache_path(config: Q1Config, *, fold: int, train_idx) -> Path | None:
    if config.artifact_dir is None or not config.cache_models:
        return None
    train_hash = _index_hash(train_idx)
    policy = config.policy
    name = (
        f"{_safe_name(config.dataset_key)}__{_safe_name(config.run_profile)}__"
        f"{_safe_name(config.mimic_mode)}__cap-{config.mimic_capacity:g}__"
        f"{_safe_name(policy.method)}-{_safe_name(policy.neighbour_mode)}-k{policy.n_neighbors}__"
        f"seed-{config.random_state}__fold-{fold}__train-{train_hash}.joblib"
    )
    return Path(config.artifact_dir) / "models" / name


def _emit_progress(progress, *, completed, total: int, started_at: float, fold: int | None, now: float | None = None) -> None:
    if progress is None:
        return
    current = time.monotonic() if now is None else now
    elapsed = current - started_at
    eta = None
    if completed is not None and completed > 0:
        eta = elapsed / completed * max(0, total - completed)
    event = {
        "completed": completed,
        "total": total,
        "fold": fold,
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


def _safe_name(value) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in str(value))


def _configure_worker_threads(worker_threads: int) -> None:
    value = str(max(1, int(worker_threads)))
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(name, value)
    try:
        import torch

        torch.set_num_threads(int(value))
    except Exception:
        pass


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
    dataset = fetch_openml(name="satimage", as_frame=True)
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
