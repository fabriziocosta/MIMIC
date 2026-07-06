"""Question 2 ADASYN-style benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
    if completed == 0:
        print(f"Starting Q2 runs: 0/{total} complete")
        return
    eta_text = f", ETA {eta}" if eta is not None else ""
    print(f"Completed run {completed}/{total}, elapsed {elapsed}{eta_text}")


@dataclass(frozen=True)
class Q2Config:
    dataset_key: str = "pima"
    run_profile: str = "view"
    random_state: int = 0
    n_runs: int = 100
    mimic_mode: str = "factorised"
    mimic_capacity_run_full: float = 0.25
    artifact_dir: str | None = None
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
    def mimic_capacity(self) -> float:
        return self.mimic_capacity_run_full

    @property
    def saves_as_profile(self) -> str:
        return "run_full" if self.run_profile == "view" else self.run_profile

    @property
    def should_run_experiment(self) -> bool:
        return self.run_profile != "view"


def q2_dataset_registry(
    *,
    artifact_dir: str | Path | None = None,
    run_profile: str = "view",
    mimic_mode: str = "factorised",
    mimic_capacity: float = 0.25,
    policy: GenerationPolicy | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    registry = pd.DataFrame(
        [
            {"key": "vehicle", "dataset": "Vehicle", "status": "optional: exact ADASYN binary source needed"},
            {"key": "pima", "dataset": "Pima Indian Diabetes", "status": "ready: OpenML data_id=37"},
            {"key": "vowel", "dataset": "Vowel recognition", "status": "optional: exact ADASYN binary source needed"},
            {"key": "ionosphere", "dataset": "Ionosphere", "status": "ready: OpenML name=ionosphere"},
            {"key": "abalone", "dataset": "Abalone", "status": "ready: OpenML data_id=183; classes 18 vs 9"},
        ]
    )
    registry["experiment"] = [
        _experiment_status(
            key,
            artifact_dir=artifact_dir,
            run_profile=run_profile,
            mimic_mode=mimic_mode,
            mimic_capacity=mimic_capacity,
            policy=policy,
            random_state=random_state,
        )
        for key in registry["key"]
    ]
    return registry


def published_reference_table() -> pd.DataFrame:
    rows = [
        ("Vehicle", "Decision tree", 0.9220, 0.8454, 0.8199, 0.8308, 0.8834),
        ("Vehicle", "SMOTE", 0.9239, 0.8236, 0.8638, 0.8418, 0.9018),
        ("Vehicle", "ADASYN", 0.9257, 0.8067, 0.9015, 0.8505, 0.9168),
        ("Pima Indian Diabetes", "Decision tree", 0.6831, 0.5460, 0.5500, 0.5469, 0.6430),
        ("Pima Indian Diabetes", "SMOTE", 0.6557, 0.5049, 0.6201, 0.5556, 0.6454),
        ("Pima Indian Diabetes", "ADASYN", 0.6837, 0.5412, 0.6097, 0.5726, 0.6625),
        ("Vowel recognition", "Decision tree", 0.9760, 0.8710, 0.8700, 0.8681, 0.9256),
        ("Vowel recognition", "SMOTE", 0.9753, 0.8365, 0.9147, 0.8717, 0.9470),
        ("Vowel recognition", "ADASYN", 0.9678, 0.7603, 0.9560, 0.8453, 0.9622),
        ("Ionosphere", "Decision tree", 0.8617, 0.8403, 0.7698, 0.8003, 0.8371),
        ("Ionosphere", "SMOTE", 0.8646, 0.8211, 0.8032, 0.8101, 0.8489),
        ("Ionosphere", "ADASYN", 0.8686, 0.8298, 0.8095, 0.8162, 0.8530),
        ("Abalone", "Decision tree", 0.9307, 0.3877, 0.2929, 0.3249, 0.5227),
        ("Abalone", "SMOTE", 0.9121, 0.2876, 0.3414, 0.3060, 0.5588),
        ("Abalone", "ADASYN", 0.8659, 0.2073, 0.4538, 0.2805, 0.6291),
    ]
    return pd.DataFrame(rows, columns=["dataset", "method", "oa", "precision", "recall", "f_measure", "g_mean"])


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


def load_q2_dataset(config: Q2Config) -> pd.DataFrame:
    key = config.dataset_key
    if key == "pima":
        return _load_pima()
    if key == "ionosphere":
        return _load_ionosphere()
    if key == "abalone":
        return _load_abalone()
    if key in {"vehicle", "vowel"}:
        raise NotImplementedError(f"{key!r} needs the exact ADASYN-paper binary source before replication")
    raise NotImplementedError(f"No Q2 loader implemented for {key!r}")


def half_split_indices(frame: pd.DataFrame, *, random_state: int, target: str = "label") -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []
    for label in ["majority", "minority"]:
        indices = frame.index[frame[target].eq(label)].to_numpy()
        shuffled = rng.permutation(indices)
        n_train = len(shuffled) // 2
        train_parts.append(shuffled[:n_train])
        test_parts.append(shuffled[n_train:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    return rng.permutation(train_idx), rng.permutation(test_idx)


def generated_count_to_balance(train: pd.DataFrame, target: str = "label") -> int:
    counts = train[target].value_counts()
    return max(0, int(counts.get("majority", 0) - counts.get("minority", 0)))


def infer_mimic_columns(frame: pd.DataFrame, target: str = "label") -> dict[str, list[str]]:
    feature_frame = frame.drop(columns=[target])
    numeric = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in feature_frame.columns if column not in numeric]
    return {"regression": numeric, "classification": categorical + [target]}


def run_q2_benchmark(frame: pd.DataFrame, config: Q2Config, progress=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    started_at = time.monotonic()
    _emit_progress(progress, completed=0, total=config.n_runs, started_at=started_at)
    for run in range(1, config.n_runs + 1):
        rows.append(evaluate_q2_run(frame, config=config, run=run))
        _emit_progress(progress, completed=run, total=config.n_runs, started_at=started_at)
    run_results = pd.DataFrame(rows)
    summary = summarize_q2_results(run_results, config)
    manuscript = manuscript_result_table(summary)
    save_q2_result_tables(config, run_results, summary, manuscript)
    return run_results, summary, manuscript


def evaluate_q2_run(frame: pd.DataFrame, *, config: Q2Config, run: int, target: str = "label") -> dict[str, float]:
    train_idx, test_idx = half_split_indices(frame, random_state=config.random_state + run, target=target)
    train = frame.loc[train_idx].reset_index(drop=True)
    test = frame.loc[test_idx].reset_index(drop=True)
    n_generated = generated_count_to_balance(train, target=target)
    augmented = augment_minority_with_mimic(train, n_generated, config=config, random_state=config.random_state + run)
    classifier = make_classifier(augmented, random_state=config.random_state + run, target=target)
    classifier.fit(augmented.drop(columns=[target]), augmented[target])
    predictions = classifier.predict(test.drop(columns=[target]))
    metrics = binary_metrics(test[target], predictions)
    return {
        "dataset_key": config.dataset_key,
        "run": run,
        "n_train": len(train),
        "n_test": len(test),
        "n_generated": n_generated,
        **metrics,
    }


def augment_minority_with_mimic(
    train: pd.DataFrame,
    n_generated: int,
    *,
    config: Q2Config,
    random_state: int,
) -> pd.DataFrame:
    if n_generated == 0:
        return train.copy()
    model = MIMIC(
        columns=infer_mimic_columns(train),
        mode=config.mimic_mode,
        capacity=config.mimic_capacity,
        policy=config.policy,
        random_state=random_state,
    )
    model.fit(train)
    synthetic = model.sample(n_generated, condition={"label": "minority"})
    return pd.concat([train, synthetic], ignore_index=True)


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
            ("classifier", DecisionTreeClassifier(random_state=random_state)),
        ]
    )


def binary_metrics(y_true, y_pred) -> dict[str, float]:
    precision, recall, f_measure, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["minority"],
        average="binary",
        pos_label="minority",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["majority", "minority"]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "oa": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
        "g_mean": float(np.sqrt(recall * specificity)),
        "specificity": specificity,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def summarize_q2_results(run_results: pd.DataFrame, config: Q2Config) -> pd.DataFrame:
    metric_columns = ["oa", "precision", "recall", "f_measure", "g_mean"]
    row = {
        "dataset_key": config.dataset_key,
        "dataset": _dataset_label(config.dataset_key),
        "method": f"MIMIC-{config.mimic_mode}",
        "run_profile": config.saves_as_profile,
        "n_runs": len(run_results),
        "artifact_dir": config.artifact_dir,
    }
    for metric in metric_columns:
        row[metric] = run_results[metric].mean()
        row[f"{metric}_std"] = run_results[metric].std(ddof=0)
    return pd.DataFrame([row])


def manuscript_result_table(summary: pd.DataFrame) -> pd.DataFrame:
    reference = published_reference_table()
    dataset = summary.loc[0, "dataset"]
    mimic = summary[["dataset", "method", "oa", "precision", "recall", "f_measure", "g_mean"]]
    return pd.concat([reference.loc[reference["dataset"].eq(dataset)], mimic], ignore_index=True)


def plot_q2_metric_comparison(manuscript_table: pd.DataFrame, *, metric: str = "g_mean", figsize=(6.5, 4.0)):
    fig, ax = plt.subplots(figsize=figsize)
    plot_frame = manuscript_table[["method", metric]].dropna()
    ax.bar(plot_frame["method"], plot_frame[metric], color=["#737373", "#4C78A8", "#F58518", "#54A24B"][: len(plot_frame)])
    ax.set_title(f"Q2 {metric.replace('_', '-')} comparison: {manuscript_table.loc[0, 'dataset']}")
    ax.set_xlabel("Method")
    ax.set_ylabel(metric.replace("_", "-"))
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig, ax


def save_q2_result_tables(
    config: Q2Config,
    run_results: pd.DataFrame,
    summary: pd.DataFrame,
    manuscript: pd.DataFrame,
) -> dict[str, Path] | None:
    paths = q2_result_table_paths(config)
    if paths is None:
        return None
    paths["run_results"].parent.mkdir(parents=True, exist_ok=True)
    run_results.to_csv(paths["run_results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    manuscript.to_csv(paths["manuscript"], index=False)
    return paths


def load_q2_result_tables(config: Q2Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = q2_result_table_paths(config)
    if paths is None:
        raise ValueError("Q2 result tables require config.artifact_dir")
    return pd.read_csv(paths["run_results"]), pd.read_csv(paths["summary"]), pd.read_csv(paths["manuscript"])


def q2_result_table_manifest(config: Q2Config) -> pd.DataFrame:
    paths = q2_result_table_paths(config)
    if paths is None:
        return pd.DataFrame(
            [{"table": "run_results", "path": None}, {"table": "summary", "path": None}, {"table": "manuscript", "path": None}]
        )
    return pd.DataFrame([{"table": name, "path": str(path)} for name, path in paths.items()])


def q2_result_table_paths(config: Q2Config) -> dict[str, Path] | None:
    if config.artifact_dir is None:
        return None
    base = Path(config.artifact_dir) / "tables" / _config_artifact_stem(config)
    return {
        "run_results": base.with_name(f"{base.name}__run_results.csv"),
        "summary": base.with_name(f"{base.name}__summary.csv"),
        "manuscript": base.with_name(f"{base.name}__manuscript.csv"),
    }


def _emit_progress(progress, *, completed: int, total: int, started_at: float, now: float | None = None) -> None:
    if progress is None:
        return
    current = time.monotonic() if now is None else now
    elapsed = current - started_at
    eta = None
    if completed > 0:
        eta = elapsed / completed * max(0, total - completed)
    progress(
        {
            "completed": completed,
            "total": total,
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
    mimic_mode: str,
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
        mimic_mode=mimic_mode,
        mimic_capacity=mimic_capacity,
        policy=selected_policy,
        random_state=random_state,
    )
    summary_path = Path(artifact_dir) / "tables" / f"{stem}__summary.csv"
    return "complete: summary csv" if summary_path.exists() else "not run"


def _config_artifact_stem(config: Q2Config) -> str:
    return _artifact_stem(
        dataset_key=config.dataset_key,
        saved_profile=config.saves_as_profile,
        mimic_mode=config.mimic_mode,
        mimic_capacity=config.mimic_capacity,
        policy=config.policy,
        random_state=config.random_state,
    )


def _artifact_stem(
    *,
    dataset_key: str,
    saved_profile: str,
    mimic_mode: str,
    mimic_capacity: float,
    policy: GenerationPolicy,
    random_state: int,
) -> str:
    return (
        f"{_safe_name(dataset_key)}__{_safe_name(saved_profile)}__"
        f"{_safe_name(mimic_mode)}__cap-{mimic_capacity:g}__"
        f"{_safe_name(policy.method)}-{_safe_name(policy.neighbour_mode)}-k{policy.n_neighbors}__"
        f"seed-{random_state}"
    )


def _dataset_label(dataset_key: str) -> str:
    registry = q2_dataset_registry()
    selected = registry.loc[registry["key"].eq(dataset_key), "dataset"]
    return selected.iloc[0] if not selected.empty else dataset_key


def _clean_column_name(column, index: int) -> str:
    name = str(column).strip().replace(" ", "_").replace("-", "_")
    return name if name else f"x{index}"


def _safe_name(value) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in str(value))


def _load_pima() -> pd.DataFrame:
    dataset = fetch_openml(data_id=37, as_frame=True)
    return standardize_binary_frame(dataset.data, dataset.target)


def _load_ionosphere() -> pd.DataFrame:
    dataset = fetch_openml(name="ionosphere", version=1, as_frame=True)
    return standardize_binary_frame(dataset.data, dataset.target)


def _load_abalone() -> pd.DataFrame:
    dataset = fetch_openml(data_id=183, as_frame=True)
    raw = dataset.frame.copy()
    target = dataset.target.name
    selected = raw.loc[raw[target].astype(str).isin(["9", "18"])].reset_index(drop=True)
    labels = selected.pop(target).astype(str)
    selected = selected.drop(columns=[column for column in ["Sex", "sex"] if column in selected.columns])
    return standardize_binary_frame(selected, labels, minority_value="18")
