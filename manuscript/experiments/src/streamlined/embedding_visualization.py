"""Reusable preparation for latent-displacement embedding visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from mimic import GenerationPolicy, MIMIC

from .config import ExperimentConfig
from .datasets import feature_roles, load_dataset


@dataclass
class LatentDisplacementEmbedding:
    dataset_key: str
    model: MIMIC
    plot_frame: pd.DataFrame
    feature_columns: list[str]
    fitted_rows: int


@dataclass(frozen=True)
class LatentDisplacementRun:
    """Configuration identifying the most recently completed embedding run."""

    config: ExperimentConfig
    sample_size: int
    seed: int
    max_plot_rows: int


def latest_latent_displacement_run_path(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / "models" / "latent_displacement" / "latest_run.joblib"


def save_latest_latent_displacement_run(run: LatentDisplacementRun) -> Path:
    path = latest_latent_displacement_run_path(run.config.artifact_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(run, path)
    return path


def load_latest_latent_displacement_run(
    artifact_dir: str | Path,
) -> LatentDisplacementRun:
    path = latest_latent_displacement_run_path(artifact_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing latest latent-displacement run manifest: {path}. "
            "Run 03_train_latent_displacement_models.ipynb first."
        )
    run = joblib.load(path)
    if not isinstance(run, LatentDisplacementRun):
        raise TypeError(f"Unexpected run manifest type at {path}: {type(run).__name__}")

    missing = [
        latent_displacement_embedding_path(
            run.config,
            dataset_key=dataset_key,
            sample_size=run.sample_size,
            seed=run.seed,
        )
        for dataset_key in run.config.profile.datasets
    ]
    missing = [model_path for model_path in missing if not model_path.exists()]
    if missing:
        formatted = ", ".join(str(model_path) for model_path in missing)
        raise FileNotFoundError(
            f"Latest latent-displacement run references missing model artifacts: {formatted}. "
            "Rerun 03_train_latent_displacement_models.ipynb."
        )
    return run


def latent_displacement_embedding_path(
    config: ExperimentConfig,
    *,
    dataset_key: str,
    sample_size: int,
    seed: int,
) -> Path:
    filename = f"{dataset_key}__n-{sample_size}__seed-{seed}.joblib"
    return config.artifact_root / "models" / "latent_displacement" / filename


def save_latent_displacement_embedding(
    view: LatentDisplacementEmbedding,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(view, output)
    return output


def load_latent_displacement_embedding(
    config: ExperimentConfig,
    *,
    dataset_key: str,
    sample_size: int,
    seed: int = 0,
) -> LatentDisplacementEmbedding:
    path = latent_displacement_embedding_path(
        config,
        dataset_key=dataset_key,
        sample_size=sample_size,
        seed=seed,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing latent-displacement embedding artifact: {path}. "
            "Run 03_train_latent_displacement_models.ipynb first."
        )
    view = joblib.load(path)
    if not isinstance(view, LatentDisplacementEmbedding):
        raise TypeError(f"Unexpected embedding artifact type at {path}: {type(view).__name__}")
    return view


def fit_and_save_latent_displacement_embeddings(
    config: ExperimentConfig,
    *,
    sample_size: int = 2000,
    seed: int = 0,
    max_plot_rows: int = 750,
) -> pd.DataFrame:
    """Fit and persist one visualization bundle for every configured dataset."""
    rows = []
    for dataset_key in config.profile.datasets:
        view = fit_latent_displacement_embedding(
            config,
            dataset_key=dataset_key,
            sample_size=sample_size,
            seed=seed,
            max_plot_rows=max_plot_rows,
        )
        path = save_latent_displacement_embedding(
            view,
            latent_displacement_embedding_path(
                config,
                dataset_key=dataset_key,
                sample_size=sample_size,
                seed=seed,
            ),
        )
        rows.append(
            {
                "dataset_key": dataset_key,
                "fitted_rows": view.fitted_rows,
                "plot_rows": len(view.plot_frame),
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def fit_latent_displacement_embedding(
    config: ExperimentConfig,
    *,
    dataset_key: str,
    sample_size: int = 2000,
    seed: int = 0,
    max_plot_rows: int = 750,
) -> LatentDisplacementEmbedding:
    """Fit one latent-displacement model on a stratified dataset sample."""
    frame = load_dataset(
        dataset_key,
        n_rows=max(1, int(sample_size)),
        random_state=seed,
    ).reset_index(drop=True)
    roles = feature_roles(frame, dataset_key)
    feature_columns = roles["numeric"] + roles["categorical"]
    policy = GenerationPolicy(
        method="displacement",
        neighbour_mode="normal",
        n_neighbors=config.n_neighbors,
        lambda_range=config.lambda_range,
    )
    model = MIMIC(
        columns={
            "ignore": roles["ignore"],
            "regression": roles["numeric"],
            "classification": roles["categorical"] + [config.target_column],
        },
        mode=config.mimic_mode,
        capacity=config.mimic_capacity,
        policy=policy,
        bootstrap=False,
        random_state=seed,
        feature_n_jobs=1,
    ).fit(frame)
    plot_rows = min(len(frame), max(1, int(max_plot_rows)))
    plot_frame = frame.sample(n=plot_rows, random_state=seed).reset_index(drop=True)
    return LatentDisplacementEmbedding(
        dataset_key=dataset_key,
        model=model,
        plot_frame=plot_frame,
        feature_columns=feature_columns,
        fitted_rows=len(frame),
    )
