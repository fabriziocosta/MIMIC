"""Question 3 Geometric-SMOTE-style benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

from joblib import Parallel, delayed, effective_n_jobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_iris, load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mimic import GenerationPolicy, MIMIC


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def print_progress_event(event: dict) -> None:
    completed = event.get("completed")
    total = event.get("total")
    elapsed = event.get("elapsed")
    eta = event.get("eta")
    parallel_workers = event.get("parallel_workers", 1)
    if completed == 0:
        print(f"Starting Q3 jobs: 0/{total} complete ({parallel_workers} worker(s))")
        return
    eta_text = f", ETA {eta}" if eta is not None else ""
    print(f"Completed job {completed}/{total}, elapsed {elapsed}{eta_text}")


@dataclass(frozen=True)
class Q3Config:
    dataset_key: str = "pima"
    run_profile: str = "view"
    random_state: int = 0
    n_repeats: int = 5
    n_splits: int = 5
    classifiers: tuple[str, ...] = ("GBC", "LR")
    mimic_modes: tuple[str, ...] = ("identity", "factorised")
    mimic_capacity_run_full: float = 0.25
    artifact_dir: str | None = None
    n_jobs: int = 1
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
    def mimic_capacity(self) -> float:
        return self.mimic_capacity_run_full

    @property
    def saves_as_profile(self) -> str:
        return "run_full" if self.run_profile == "view" else self.run_profile

    @property
    def should_run_experiment(self) -> bool:
        return self.run_profile != "view"


def q3_dataset_registry(
    *,
    artifact_dir: str | Path | None = None,
    run_profile: str = "view",
    mimic_modes: tuple[str, ...] = ("identity", "factorised"),
    mimic_capacity: float = 0.25,
    policy: GenerationPolicy | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    registry = pd.DataFrame(
        [
            {"key": "breast", "dataset": "Breast", "features": 9, "instances": 106, "minority": 36, "majority": 70, "ir": 1.94, "status": "optional: exact UCI source mapping needed"},
            {"key": "ecoli", "dataset": "Ecoli", "features": 7, "instances": 336, "minority": 52, "majority": 284, "ir": 5.46, "status": "optional: exact class mapping needed"},
            {"key": "eucalyptus", "dataset": "Eucalyptus", "features": 8, "instances": 642, "minority": 98, "majority": 544, "ir": 5.55, "status": "optional: source needed"},
            {"key": "glass", "dataset": "Glass", "features": 9, "instances": 214, "minority": 70, "majority": 144, "ir": 2.06, "status": "optional: exact class mapping needed"},
            {"key": "haberman", "dataset": "Haberman", "features": 3, "instances": 306, "minority": 81, "majority": 225, "ir": 2.78, "status": "optional: exact class mapping needed"},
            {"key": "heart", "dataset": "Heart", "features": 13, "instances": 270, "minority": 120, "majority": 150, "ir": 1.25, "status": "optional: exact source mapping needed"},
            {"key": "iris", "dataset": "Iris", "features": 4, "instances": 150, "minority": 50, "majority": 100, "ir": 2.00, "status": "ready: sklearn iris class 0 vs rest"},
            {"key": "libra", "dataset": "Libra", "features": 90, "instances": 360, "minority": 72, "majority": 288, "ir": 4.00, "status": "optional: exact class mapping needed"},
            {"key": "liver", "dataset": "Liver", "features": 6, "instances": 345, "minority": 145, "majority": 200, "ir": 1.38, "status": "optional: exact source mapping needed"},
            {"key": "pima", "dataset": "Pima", "features": 8, "instances": 768, "minority": 268, "majority": 500, "ir": 1.87, "status": "ready: OpenML data_id=37"},
            {"key": "segment", "dataset": "Segment", "features": 16, "instances": 2310, "minority": 330, "majority": 1980, "ir": 6.00, "status": "optional: exact class mapping needed"},
            {"key": "vehicle", "dataset": "Vehicle", "features": 18, "instances": 846, "minority": 199, "majority": 647, "ir": 3.25, "status": "optional: exact class mapping needed"},
            {"key": "wine", "dataset": "Wine", "features": 13, "instances": 178, "minority": 71, "majority": 107, "ir": 1.51, "status": "ready: sklearn wine class 1 vs rest"},
        ]
    )
    registry["experiment"] = [
        _experiment_status(
            key,
            artifact_dir=artifact_dir,
            run_profile=run_profile,
            mimic_modes=mimic_modes,
            mimic_capacity=mimic_capacity,
            policy=policy,
            random_state=random_state,
        )
        for key in registry["key"]
    ]
    return registry


def published_reference_table() -> pd.DataFrame:
    rows = [
        ("Breast", "GBC", "F", 0.700, 0.715, 0.727, 0.705, 0.673, 0.722, "Borderline SMOTE1", 0.727),
        ("Breast", "GBC", "G", 0.770, 0.780, 0.795, 0.774, 0.748, 0.789, "Borderline SMOTE1", 0.795),
        ("Breast", "GBC", "AUC", 0.867, 0.867, 0.867, 0.869, 0.858, 0.874, "Geometric SMOTE", 0.874),
        ("Breast", "LR", "F", 0.687, 0.732, 0.747, 0.734, 0.730, 0.748, "Geometric SMOTE", 0.748),
        ("Breast", "LR", "G", 0.752, 0.799, 0.812, 0.803, 0.798, 0.812, "Borderline SMOTE1 / Geometric SMOTE", 0.812),
        ("Breast", "LR", "AUC", 0.884, 0.890, 0.888, 0.889, 0.889, 0.896, "Geometric SMOTE", 0.896),
        ("Ecoli", "GBC", "F", 0.781, 0.777, 0.773, 0.770, 0.662, 0.821, "Geometric SMOTE", 0.821),
        ("Ecoli", "GBC", "G", 0.868, 0.872, 0.868, 0.862, 0.861, 0.904, "Geometric SMOTE", 0.904),
        ("Ecoli", "GBC", "AUC", 0.948, 0.945, 0.945, 0.940, 0.937, 0.960, "Geometric SMOTE", 0.960),
        ("Ecoli", "LR", "F", 0.249, 0.716, 0.682, 0.641, 0.473, 0.718, "Geometric SMOTE", 0.718),
        ("Ecoli", "LR", "G", 0.378, 0.894, 0.885, 0.870, 0.759, 0.900, "Geometric SMOTE", 0.900),
        ("Ecoli", "LR", "AUC", 0.934, 0.935, 0.929, 0.924, 0.896, 0.936, "Geometric SMOTE", 0.936),
        ("Eucalyptus", "GBC", "F", 0.551, 0.526, 0.540, 0.535, 0.506, 0.549, "No oversampling", 0.551),
        ("Eucalyptus", "GBC", "G", 0.682, 0.699, 0.703, 0.714, 0.688, 0.730, "Geometric SMOTE", 0.730),
        ("Eucalyptus", "GBC", "AUC", 0.878, 0.869, 0.872, 0.870, 0.841, 0.874, "No oversampling", 0.878),
        ("Eucalyptus", "LR", "F", 0.207, 0.529, 0.530, 0.513, 0.468, 0.540, "Geometric SMOTE", 0.540),
        ("Eucalyptus", "LR", "G", 0.399, 0.790, 0.787, 0.785, 0.757, 0.797, "Geometric SMOTE", 0.797),
        ("Eucalyptus", "LR", "AUC", 0.857, 0.886, 0.883, 0.879, 0.870, 0.890, "Geometric SMOTE", 0.890),
        ("Glass", "GBC", "F", 0.779, 0.774, 0.774, 0.774, 0.758, 0.792, "Geometric SMOTE", 0.792),
        ("Glass", "GBC", "G", 0.831, 0.830, 0.833, 0.836, 0.822, 0.846, "Geometric SMOTE", 0.846),
        ("Glass", "GBC", "AUC", 0.924, 0.920, 0.918, 0.919, 0.911, 0.929, "Geometric SMOTE", 0.929),
        ("Glass", "LR", "F", 0.505, 0.657, 0.648, 0.651, 0.650, 0.661, "Geometric SMOTE", 0.661),
        ("Glass", "LR", "G", 0.613, 0.733, 0.715, 0.714, 0.719, 0.734, "Geometric SMOTE", 0.734),
        ("Glass", "LR", "AUC", 0.824, 0.825, 0.814, 0.818, 0.819, 0.825, "SMOTE / Geometric SMOTE", 0.825),
        ("Haberman", "GBC", "F", 0.339, 0.393, 0.388, 0.385, 0.376, 0.426, "Geometric SMOTE", 0.426),
        ("Haberman", "GBC", "G", 0.505, 0.558, 0.551, 0.549, 0.542, 0.584, "Geometric SMOTE", 0.584),
        ("Haberman", "GBC", "AUC", 0.627, 0.637, 0.641, 0.636, 0.619, 0.668, "Geometric SMOTE", 0.668),
        ("Haberman", "LR", "F", 0.239, 0.477, 0.485, 0.481, 0.418, 0.484, "Borderline SMOTE1", 0.485),
        ("Haberman", "LR", "G", 0.376, 0.622, 0.632, 0.628, 0.578, 0.628, "Borderline SMOTE1", 0.632),
        ("Haberman", "LR", "AUC", 0.686, 0.693, 0.688, 0.684, 0.650, 0.694, "Geometric SMOTE", 0.694),
        ("Heart", "GBC", "F", 0.745, 0.750, 0.753, 0.754, 0.739, 0.761, "Geometric SMOTE", 0.761),
        ("Heart", "GBC", "G", 0.771, 0.775, 0.776, 0.779, 0.765, 0.781, "Geometric SMOTE", 0.781),
        ("Heart", "GBC", "AUC", 0.859, 0.862, 0.864, 0.864, 0.856, 0.868, "Geometric SMOTE", 0.868),
        ("Heart", "LR", "F", 0.822, 0.822, 0.823, 0.824, 0.820, 0.826, "Geometric SMOTE", 0.826),
        ("Heart", "LR", "G", 0.840, 0.840, 0.841, 0.841, 0.837, 0.843, "Geometric SMOTE", 0.843),
        ("Heart", "LR", "AUC", 0.902, 0.903, 0.901, 0.900, 0.901, 0.903, "SMOTE / Geometric SMOTE", 0.903),
        ("Iris", "GBC", "F", 0.923, 0.937, 0.915, 0.922, 0.911, 0.939, "Geometric SMOTE", 0.939),
        ("Iris", "GBC", "G", 0.944, 0.954, 0.939, 0.945, 0.940, 0.956, "Geometric SMOTE", 0.956),
        ("Iris", "GBC", "AUC", 0.980, 0.985, 0.978, 0.984, 0.972, 0.988, "Geometric SMOTE", 0.988),
        ("Iris", "LR", "F", 0.334, 0.648, 0.637, 0.636, 0.674, 0.657, "ADASYN", 0.674),
        ("Iris", "LR", "G", 0.453, 0.731, 0.715, 0.712, 0.752, 0.739, "ADASYN", 0.752),
        ("Iris", "LR", "AUC", 0.793, 0.792, 0.747, 0.745, 0.803, 0.801, "ADASYN", 0.803),
        ("Libra", "GBC", "F", 0.780, 0.850, 0.839, 0.876, 0.786, 0.924, "Geometric SMOTE", 0.924),
        ("Libra", "GBC", "G", 0.821, 0.890, 0.875, 0.915, 0.847, 0.946, "Geometric SMOTE", 0.946),
        ("Libra", "GBC", "AUC", 0.943, 0.969, 0.962, 0.976, 0.931, 0.987, "Geometric SMOTE", 0.987),
        ("Libra", "LR", "F", 0.316, 0.575, 0.512, 0.475, 0.631, 0.565, "ADASYN", 0.631),
        ("Libra", "LR", "G", 0.432, 0.745, 0.686, 0.659, 0.779, 0.738, "ADASYN", 0.779),
        ("Libra", "LR", "AUC", 0.737, 0.766, 0.733, 0.708, 0.776, 0.765, "ADASYN", 0.776),
        ("Liver", "GBC", "F", 0.636, 0.630, 0.639, 0.644, 0.642, 0.657, "Geometric SMOTE", 0.657),
        ("Liver", "GBC", "G", 0.688, 0.681, 0.687, 0.693, 0.686, 0.701, "Geometric SMOTE", 0.701),
        ("Liver", "GBC", "AUC", 0.753, 0.752, 0.755, 0.754, 0.756, 0.758, "Geometric SMOTE", 0.758),
        ("Liver", "LR", "F", 0.579, 0.633, 0.622, 0.629, 0.626, 0.640, "Geometric SMOTE", 0.640),
        ("Liver", "LR", "G", 0.644, 0.669, 0.658, 0.661, 0.650, 0.669, "SMOTE / Geometric SMOTE", 0.669),
        ("Liver", "LR", "AUC", 0.713, 0.712, 0.707, 0.709, 0.710, 0.715, "Geometric SMOTE", 0.715),
        ("Pima", "GBC", "F", 0.623, 0.651, 0.654, 0.652, 0.644, 0.659, "Geometric SMOTE", 0.659),
        ("Pima", "GBC", "G", 0.704, 0.728, 0.731, 0.729, 0.722, 0.734, "Geometric SMOTE", 0.734),
        ("Pima", "GBC", "AUC", 0.812, 0.811, 0.806, 0.806, 0.800, 0.822, "Geometric SMOTE", 0.822),
        ("Pima", "LR", "F", 0.617, 0.675, 0.675, 0.674, 0.676, 0.677, "Geometric SMOTE", 0.677),
        ("Pima", "LR", "G", 0.692, 0.748, 0.747, 0.746, 0.747, 0.749, "Geometric SMOTE", 0.749),
        ("Pima", "LR", "AUC", 0.825, 0.827, 0.825, 0.824, 0.824, 0.830, "Geometric SMOTE", 0.830),
        ("Segment", "GBC", "F", 0.925, 0.927, 0.912, 0.887, 0.856, 0.934, "Geometric SMOTE", 0.934),
        ("Segment", "GBC", "G", 0.941, 0.967, 0.961, 0.960, 0.956, 0.970, "Geometric SMOTE", 0.970),
        ("Segment", "GBC", "AUC", 0.996, 0.997, 0.995, 0.994, 0.990, 0.997, "SMOTE / Geometric SMOTE", 0.997),
        ("Segment", "LR", "F", 0.648, 0.645, 0.634, 0.622, 0.552, 0.640, "No oversampling", 0.648),
        ("Segment", "LR", "G", 0.749, 0.881, 0.881, 0.888, 0.850, 0.881, "Borderline SMOTE2", 0.888),
        ("Segment", "LR", "AUC", 0.942, 0.942, 0.930, 0.923, 0.922, 0.944, "Geometric SMOTE", 0.944),
        ("Vehicle", "GBC", "F", 0.923, 0.927, 0.929, 0.925, 0.923, 0.935, "Geometric SMOTE", 0.935),
        ("Vehicle", "GBC", "G", 0.951, 0.958, 0.958, 0.961, 0.962, 0.969, "Geometric SMOTE", 0.969),
        ("Vehicle", "GBC", "AUC", 0.994, 0.994, 0.995, 0.994, 0.993, 0.995, "Borderline SMOTE1 / Geometric SMOTE", 0.995),
        ("Vehicle", "LR", "F", 0.940, 0.941, 0.940, 0.895, 0.816, 0.940, "SMOTE", 0.941),
        ("Vehicle", "LR", "G", 0.961, 0.968, 0.967, 0.958, 0.924, 0.969, "Geometric SMOTE", 0.969),
        ("Vehicle", "LR", "AUC", 0.995, 0.995, 0.995, 0.992, 0.983, 0.995, "No oversampling / SMOTE / Borderline SMOTE1 / Geometric SMOTE", 0.995),
        ("Wine", "GBC", "F", 0.911, 0.908, 0.912, 0.907, 0.875, 0.933, "Geometric SMOTE", 0.933),
        ("Wine", "GBC", "G", 0.925, 0.923, 0.927, 0.922, 0.898, 0.943, "Geometric SMOTE", 0.943),
        ("Wine", "GBC", "AUC", 0.979, 0.978, 0.978, 0.975, 0.965, 0.986, "Geometric SMOTE", 0.986),
        ("Wine", "LR", "F", 0.928, 0.935, 0.939, 0.937, 0.898, 0.938, "Borderline SMOTE1", 0.939),
        ("Wine", "LR", "G", 0.941, 0.947, 0.951, 0.950, 0.918, 0.951, "Borderline SMOTE1 / Geometric SMOTE", 0.951),
        ("Wine", "LR", "AUC", 0.992, 0.992, 0.991, 0.992, 0.985, 0.993, "Geometric SMOTE", 0.993),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "classifier",
            "metric",
            "no_oversampling",
            "smote",
            "borderline_smote1",
            "borderline_smote2",
            "adasyn",
            "geometric_smote",
            "published_best_method",
            "published_best_value",
        ],
    )


def load_q3_dataset(config: Q3Config) -> pd.DataFrame:
    if config.dataset_key == "pima":
        dataset = fetch_openml(data_id=37, as_frame=True)
        return standardize_binary_frame(dataset.data, dataset.target)
    if config.dataset_key == "iris":
        dataset = load_iris(as_frame=True)
        return standardize_binary_frame(dataset.data, dataset.target, minority_value=0)
    if config.dataset_key == "wine":
        dataset = load_wine(as_frame=True)
        return standardize_binary_frame(dataset.data, dataset.target, minority_value=1)
    raise NotImplementedError(f"{config.dataset_key!r} needs the exact Geometric-SMOTE dataset source and class mapping")


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


def generated_count_to_balance(train: pd.DataFrame, target: str = "label") -> int:
    counts = train[target].value_counts()
    return max(0, int(counts.get("majority", 0) - counts.get("minority", 0)))


def infer_mimic_columns(frame: pd.DataFrame, target: str = "label") -> dict[str, list[str]]:
    feature_frame = frame.drop(columns=[target])
    numeric = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in feature_frame.columns if column not in numeric]
    return {"regression": numeric, "classification": categorical + [target]}


def run_q3_benchmark(frame: pd.DataFrame, config: Q3Config, progress=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_results = run_q3_jobs(frame, config, progress=progress)
    summary = summarize_q3_results(fold_results, config)
    manuscript = manuscript_result_table(summary, dataset_key=config.dataset_key)
    save_q3_result_tables(config, fold_results, summary, manuscript)
    return fold_results, summary, manuscript


def run_q3_jobs(frame: pd.DataFrame, config: Q3Config, progress=None) -> pd.DataFrame:
    jobs = q3_jobs(frame, config)
    total = len(jobs)
    started_at = time.monotonic()
    parallel_workers = _effective_worker_count(config.n_jobs, total)
    _emit_progress(progress, completed=0, total=total, started_at=started_at, parallel_workers=parallel_workers)
    if config.n_jobs == 1:
        results = []
        for completed, job in enumerate(jobs, start=1):
            results.append(evaluate_q3_fold(frame, config=config, **job))
            _emit_progress(progress, completed=completed, total=total, started_at=started_at, parallel_workers=parallel_workers)
    else:
        task_iter = (delayed(evaluate_q3_fold)(frame, config=config, **job) for job in jobs)
        try:
            result_iter = Parallel(n_jobs=config.n_jobs, return_as="generator_unordered")(task_iter)
            results = []
            for result in result_iter:
                results.append(result)
                _emit_progress(progress, completed=len(results), total=total, started_at=started_at, parallel_workers=parallel_workers)
        except TypeError:
            task_iter = (delayed(evaluate_q3_fold)(frame, config=config, **job) for job in jobs)
            results = Parallel(n_jobs=config.n_jobs)(task_iter)
            _emit_progress(progress, completed=total, total=total, started_at=started_at, parallel_workers=parallel_workers)
    return pd.DataFrame(results)


def q3_jobs(frame: pd.DataFrame, config: Q3Config, target: str = "label") -> list[dict]:
    jobs = []
    for repeat in range(1, config.n_repeats + 1):
        splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state + repeat)
        X = frame.drop(columns=[target])
        y = frame[target]
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
            for mimic_mode in config.mimic_modes:
                for classifier_name in config.classifiers:
                    for params in classifier_grid(classifier_name):
                        jobs.append(
                            {
                                "repeat": repeat,
                                "fold": fold,
                                "train_idx": train_idx,
                                "test_idx": test_idx,
                                "mimic_mode": mimic_mode,
                                "classifier_name": classifier_name,
                                "classifier_params": params,
                            }
                        )
    return jobs


def evaluate_q3_fold(
    frame: pd.DataFrame,
    *,
    config: Q3Config,
    repeat: int,
    fold: int,
    train_idx,
    test_idx,
    mimic_mode: str,
    classifier_name: str,
    classifier_params: dict,
    target: str = "label",
) -> dict:
    _configure_worker_threads(config.worker_threads)
    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    n_generated = generated_count_to_balance(train, target=target)
    augmented = augment_minority_with_mimic(
        train,
        n_generated,
        config=config,
        mimic_mode=mimic_mode,
        random_state=config.random_state + repeat * 1000 + fold,
    )
    classifier = make_classifier(
        classifier_name,
        classifier_params,
        train=augmented,
        random_state=config.random_state + repeat * 1000 + fold,
        target=target,
    )
    classifier.fit(augmented.drop(columns=[target]), augmented[target])
    predictions = classifier.predict(test.drop(columns=[target]))
    positive_class_index = list(classifier.classes_).index("minority")
    scores = classifier.predict_proba(test.drop(columns=[target]))[:, positive_class_index]
    metrics = binary_metrics(test[target], predictions, scores)
    return {
        "dataset_key": config.dataset_key,
        "repeat": repeat,
        "fold": fold,
        "mimic_mode": mimic_mode,
        "classifier": classifier_name,
        "classifier_params": repr(classifier_params),
        "n_train": len(train),
        "n_test": len(test),
        "n_generated": n_generated,
        **metrics,
    }


def augment_minority_with_mimic(
    train: pd.DataFrame,
    n_generated: int,
    *,
    config: Q3Config,
    mimic_mode: str,
    random_state: int,
) -> pd.DataFrame:
    if n_generated == 0:
        return train.copy()
    model = MIMIC(
        columns=infer_mimic_columns(train),
        mode=mimic_mode,
        capacity=config.mimic_capacity,
        policy=config.policy,
        random_state=random_state,
        feature_n_jobs=config.mimic_feature_n_jobs,
    )
    model.fit(train)
    synthetic = model.sample(n_generated, condition={"label": "minority"})
    return pd.concat([train, synthetic], ignore_index=True)


def classifier_grid(classifier_name: str) -> list[dict]:
    if classifier_name == "GBC":
        return [{"max_depth": depth, "n_estimators": n_estimators} for depth in (5, 8) for n_estimators in (50, 100)]
    if classifier_name == "LR":
        return [{"max_iter": 1000, "solver": "lbfgs"}]
    raise ValueError(f"Unknown Q3 classifier {classifier_name!r}")


def make_classifier(
    classifier_name: str,
    classifier_params: dict,
    *,
    train: pd.DataFrame,
    random_state: int,
    target: str = "label",
) -> Pipeline:
    X = train.drop(columns=[target])
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )
    if classifier_name == "GBC":
        classifier = GradientBoostingClassifier(random_state=random_state, **classifier_params)
    elif classifier_name == "LR":
        classifier = LogisticRegression(random_state=random_state, **classifier_params)
    else:
        raise ValueError(f"Unknown Q3 classifier {classifier_name!r}")
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def binary_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    y_true_binary = pd.Series(y_true).eq("minority").astype(int).to_numpy()
    y_pred_binary = pd.Series(y_pred).eq("minority").astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["majority", "minority"]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "f_measure": f1_score(y_true_binary, y_pred_binary, zero_division=0),
        "g_mean": float(np.sqrt(recall * specificity)),
        "auc": roc_auc_score(y_true_binary, y_score),
        "specificity": specificity,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def summarize_q3_results(fold_results: pd.DataFrame, config: Q3Config) -> pd.DataFrame:
    rows = []
    metric_map = {"F": "f_measure", "G": "g_mean", "AUC": "auc"}
    for (mimic_mode, classifier, classifier_params), group in fold_results.groupby(
        ["mimic_mode", "classifier", "classifier_params"], sort=False
    ):
        for metric, column in metric_map.items():
            rows.append(
                {
                    "dataset_key": config.dataset_key,
                    "dataset": _dataset_label(config.dataset_key),
                    "run_profile": config.saves_as_profile,
                    "mimic_mode": mimic_mode,
                    "method": f"MIMIC-{mimic_mode}",
                    "classifier": classifier,
                    "classifier_params": classifier_params,
                    "metric": metric,
                    "value": group[column].mean(),
                    "std": group[column].std(ddof=0),
                    "n_folds": len(group),
                    "artifact_dir": config.artifact_dir,
                }
            )
    summary = pd.DataFrame(rows)
    return (
        summary.sort_values(["mimic_mode", "classifier", "metric", "value"], ascending=[True, True, True, False])
        .groupby(["mimic_mode", "classifier", "metric"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def manuscript_result_table(summary: pd.DataFrame, *, dataset_key: str) -> pd.DataFrame:
    dataset = _dataset_label(dataset_key)
    reference = published_reference_table().loc[published_reference_table()["dataset"].eq(dataset)].copy()
    for mode in ["identity", "factorised"]:
        values = summary.loc[summary["mimic_mode"].eq(mode), ["classifier", "metric", "value"]]
        mapping = {(row.classifier, row.metric): row.value for row in values.itertuples(index=False)}
        reference[f"mimic_{mode}"] = [mapping.get((row.classifier, row.metric), np.nan) for row in reference.itertuples()]
        reference[f"mimic_{mode}_delta_vs_best"] = reference[f"mimic_{mode}"] - reference["published_best_value"]
    return reference.reset_index(drop=True)


def plot_q3_metric_comparison(manuscript_table: pd.DataFrame, *, classifier: str = "GBC", metric: str = "AUC", figsize=(8.0, 4.0)):
    row = manuscript_table.loc[manuscript_table["classifier"].eq(classifier) & manuscript_table["metric"].eq(metric)]
    if row.empty:
        raise ValueError(f"No Q3 row for classifier={classifier!r}, metric={metric!r}")
    row = row.iloc[0]
    labels = ["No oversampling", "SMOTE", "B-SMOTE1", "B-SMOTE2", "ADASYN", "G-SMOTE", "MIMIC identity", "MIMIC factorised"]
    values = [
        row["no_oversampling"],
        row["smote"],
        row["borderline_smote1"],
        row["borderline_smote2"],
        row["adasyn"],
        row["geometric_smote"],
        row["mimic_identity"],
        row["mimic_factorised"],
    ]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values, color=["#737373", "#4C78A8", "#F58518", "#B279A2", "#54A24B", "#E45756", "#72B7B2", "#EECA3B"])
    ax.set_title(f"Q3 {metric} comparison: {row['dataset']} / {classifier}")
    ax.set_xlabel("Method")
    ax.set_ylabel(metric)
    ax.autoscale(axis="y")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig, ax


def save_q3_result_tables(
    config: Q3Config,
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
    manuscript: pd.DataFrame,
) -> dict[str, Path] | None:
    paths = q3_result_table_paths(config)
    if paths is None:
        return None
    paths["fold_results"].parent.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(paths["fold_results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    manuscript.to_csv(paths["manuscript"], index=False)
    return paths


def load_q3_result_tables(config: Q3Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = q3_result_table_paths(config)
    if paths is None:
        raise ValueError("Q3 result tables require config.artifact_dir")
    return pd.read_csv(paths["fold_results"]), pd.read_csv(paths["summary"]), pd.read_csv(paths["manuscript"])


def q3_result_table_manifest(config: Q3Config) -> pd.DataFrame:
    paths = q3_result_table_paths(config)
    if paths is None:
        return pd.DataFrame(
            [{"table": "fold_results", "path": None}, {"table": "summary", "path": None}, {"table": "manuscript", "path": None}]
        )
    return pd.DataFrame([{"table": name, "path": str(path)} for name, path in paths.items()])


def q3_result_table_paths(config: Q3Config) -> dict[str, Path] | None:
    if config.artifact_dir is None:
        return None
    base = Path(config.artifact_dir) / "tables" / _config_artifact_stem(config)
    return {
        "fold_results": base.with_name(f"{base.name}__fold_results.csv"),
        "summary": base.with_name(f"{base.name}__summary.csv"),
        "manuscript": base.with_name(f"{base.name}__manuscript.csv"),
    }


def _emit_progress(progress, *, completed: int, total: int, started_at: float, parallel_workers: int, now: float | None = None) -> None:
    if progress is None:
        return
    current = time.monotonic() if now is None else now
    elapsed = current - started_at
    eta = elapsed / completed * max(0, total - completed) if completed > 0 else None
    progress(
        {
            "completed": completed,
            "total": total,
            "parallel_workers": parallel_workers,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "elapsed": format_seconds(elapsed),
            "eta": format_seconds(eta) if eta is not None else None,
        }
    )


def _experiment_status(
    dataset_key: str,
    *,
    artifact_dir: str | Path | None,
    run_profile: str,
    mimic_modes: tuple[str, ...],
    mimic_capacity: float,
    policy: GenerationPolicy | None,
    random_state: int,
) -> str:
    if artifact_dir is None:
        return "unknown: no artifact_dir"
    selected_policy = policy or GenerationPolicy(method="smote", neighbour_mode="normal", n_neighbors=5)
    saved_profile = "run_full" if run_profile == "view" else run_profile
    stem = _artifact_stem(
        dataset_key=dataset_key,
        saved_profile=saved_profile,
        mimic_modes=mimic_modes,
        mimic_capacity=mimic_capacity,
        policy=selected_policy,
        random_state=random_state,
    )
    summary_path = Path(artifact_dir) / "tables" / f"{stem}__summary.csv"
    return "complete: summary csv" if summary_path.exists() else "not run"


def _config_artifact_stem(config: Q3Config) -> str:
    return _artifact_stem(
        dataset_key=config.dataset_key,
        saved_profile=config.saves_as_profile,
        mimic_modes=config.mimic_modes,
        mimic_capacity=config.mimic_capacity,
        policy=config.policy,
        random_state=config.random_state,
    )


def _artifact_stem(
    *,
    dataset_key: str,
    saved_profile: str,
    mimic_modes: tuple[str, ...],
    mimic_capacity: float,
    policy: GenerationPolicy,
    random_state: int,
) -> str:
    modes = "-".join(_safe_name(mode) for mode in mimic_modes)
    return (
        f"{_safe_name(dataset_key)}__{_safe_name(saved_profile)}__"
        f"{modes}__cap-{mimic_capacity:g}__"
        f"{_safe_name(policy.method)}-{_safe_name(policy.neighbour_mode)}-k{policy.n_neighbors}__"
        f"seed-{random_state}"
    )


def _dataset_label(dataset_key: str) -> str:
    registry = q3_dataset_registry()
    selected = registry.loc[registry["key"].eq(dataset_key), "dataset"]
    return selected.iloc[0] if not selected.empty else dataset_key


def _effective_worker_count(n_jobs: int, total: int) -> int:
    if total <= 0:
        return 0
    try:
        workers = effective_n_jobs(n_jobs)
    except Exception:
        workers = 1
    return max(1, min(total, workers))


def _configure_worker_threads(worker_threads: int) -> None:
    import os

    threads = str(max(1, int(worker_threads)))
    os.environ.setdefault("OMP_NUM_THREADS", threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", threads)
    os.environ.setdefault("MKL_NUM_THREADS", threads)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", threads)


def _clean_column_name(column, index: int) -> str:
    name = str(column).strip().replace(" ", "_").replace("-", "_")
    return name if name else f"x{index}"


def _safe_name(value) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in str(value))
