"""Training-set construction and synthetic sampling backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mimic import GenerationPolicy, MIMIC

from .datasets import feature_roles
from .preprocessing import PreparedData, transform_frame


@dataclass
class BalancedTrainingSet:
    X: np.ndarray
    y: np.ndarray
    generated_count: int
    real_minority_count: int
    real_majority_count: int


def make_imbalanced_subset(train: pd.DataFrame, *, ratio: float, training_size: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    majority = train.loc[train["label"].eq("majority")]
    minority = train.loc[train["label"].eq("minority")]
    if majority.empty or minority.empty:
        raise ValueError("Both majority and minority rows are required")
    minority_target = max(2, int(round(training_size / (ratio + 1.0))))
    majority_target = max(2, int(round(minority_target * ratio)))
    minority_target = min(minority_target, len(minority))
    majority_target = min(majority_target, len(majority))
    minority_idx = rng.choice(minority.index.to_numpy(), size=minority_target, replace=False)
    majority_idx = rng.choice(majority.index.to_numpy(), size=majority_target, replace=False)
    subset = pd.concat([majority.loc[majority_idx], minority.loc[minority_idx]], axis=0)
    return subset.sample(frac=1.0, random_state=random_state)


def build_balanced_training_set(
    method: str,
    *,
    imbalanced_raw: pd.DataFrame,
    train_pool_raw: pd.DataFrame | None = None,
    prepared: PreparedData,
    dataset_key: str,
    random_state: int,
    n_neighbors: int,
    lambda_range: tuple[float, float],
    mimic_mode: str,
    mimic_capacity: float,
    repair_direct_samples: bool = True,
) -> BalancedTrainingSet:
    y_raw = imbalanced_raw["label"].eq("minority").astype(int).to_numpy()
    X_raw = transform_frame(
        prepared.preprocessor,
        imbalanced_raw,
        numeric_columns=prepared.numeric_columns,
        categorical_columns=prepared.categorical_columns,
    )
    minority_pos = np.flatnonzero(y_raw == 1)
    majority_pos = np.flatnonzero(y_raw == 0)
    deficit = max(0, len(majority_pos) - len(minority_pos))
    if method == "real_balanced":
        return _real_balanced(
            X_raw,
            y_raw,
            minority_pos,
            majority_pos,
            imbalanced_raw=imbalanced_raw,
            train_pool_raw=train_pool_raw,
            prepared=prepared,
            random_state=random_state,
        )
    if deficit == 0:
        return BalancedTrainingSet(X=X_raw, y=y_raw, generated_count=0, real_minority_count=len(minority_pos), real_majority_count=len(majority_pos))
    if method == "direct_smote":
        synthetic = _direct_generate(X_raw, minority_pos, deficit, method="smote", n_neighbors=n_neighbors, lambda_range=lambda_range, random_state=random_state)
        if repair_direct_samples:
            synthetic = repair_preprocessed_samples(synthetic, prepared)
    elif method == "direct_displacement":
        synthetic = _direct_generate(X_raw, minority_pos, deficit, method="displacement", n_neighbors=n_neighbors, lambda_range=lambda_range, random_state=random_state)
        if repair_direct_samples:
            synthetic = repair_preprocessed_samples(synthetic, prepared)
    elif method in {"latent_smote", "latent_displacement"}:
        synthetic = _latent_generate(
            method,
            imbalanced_raw=imbalanced_raw,
            prepared=prepared,
            dataset_key=dataset_key,
            n_samples=deficit,
            random_state=random_state,
            n_neighbors=n_neighbors,
            lambda_range=lambda_range,
            mimic_mode=mimic_mode,
            mimic_capacity=mimic_capacity,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")
    X = np.vstack([X_raw, synthetic])
    y = np.concatenate([y_raw, np.ones(deficit, dtype=int)])
    return BalancedTrainingSet(X=X, y=y, generated_count=deficit, real_minority_count=len(minority_pos), real_majority_count=len(majority_pos))


def generated_count_for_balance(labels: pd.Series | np.ndarray) -> int:
    arr = np.asarray(labels)
    majority = int(np.sum(arr == "majority")) if arr.dtype.kind in {"O", "U", "S"} else int(np.sum(arr == 0))
    minority = int(np.sum(arr == "minority")) if arr.dtype.kind in {"O", "U", "S"} else int(np.sum(arr == 1))
    return max(0, majority - minority)


def repair_preprocessed_samples(X: np.ndarray, prepared: PreparedData) -> np.ndarray:
    repaired = np.asarray(X, dtype=float).copy()
    for sl in prepared.categorical_slices.values():
        block = repaired[:, sl]
        if block.shape[1] == 0:
            continue
        winners = np.argmax(block, axis=1)
        block[:] = 0.0
        block[np.arange(block.shape[0]), winners] = 1.0
        repaired[:, sl] = block
    return repaired


def _real_balanced(
    X: np.ndarray,
    y: np.ndarray,
    minority_pos: np.ndarray,
    majority_pos: np.ndarray,
    *,
    imbalanced_raw: pd.DataFrame,
    train_pool_raw: pd.DataFrame | None,
    prepared: PreparedData,
    random_state: int,
) -> BalancedTrainingSet:
    rng = np.random.default_rng(random_state)
    deficit = max(0, len(majority_pos) - len(minority_pos))
    if deficit == 0:
        return BalancedTrainingSet(X=X, y=y, generated_count=0, real_minority_count=len(minority_pos), real_majority_count=len(majority_pos))

    extra_minority = _available_extra_minority(imbalanced_raw, train_pool_raw, random_state=random_state)
    extra_needed = min(deficit, len(extra_minority))
    extra_X = (
        transform_frame(
            prepared.preprocessor,
            extra_minority.iloc[:extra_needed],
            numeric_columns=prepared.numeric_columns,
            categorical_columns=prepared.categorical_columns,
        )
        if extra_needed
        else np.empty((0, X.shape[1]))
    )
    total_minority = len(minority_pos) + extra_needed
    if total_minority >= len(majority_pos):
        X_balanced = np.vstack([X, extra_X])
        y_balanced = np.concatenate([y, np.ones(extra_needed, dtype=int)])
        return BalancedTrainingSet(
            X=X_balanced,
            y=y_balanced,
            generated_count=0,
            real_minority_count=total_minority,
            real_majority_count=len(majority_pos),
        )

    selected_majority = rng.choice(majority_pos, size=total_minority, replace=False)
    selected = np.concatenate([selected_majority, minority_pos])
    rng.shuffle(selected)
    X_balanced = np.vstack([X[selected], extra_X])
    y_balanced = np.concatenate([y[selected], np.ones(extra_needed, dtype=int)])
    return BalancedTrainingSet(
        X=X_balanced,
        y=y_balanced,
        generated_count=0,
        real_minority_count=total_minority,
        real_majority_count=total_minority,
    )


def _available_extra_minority(imbalanced_raw: pd.DataFrame, train_pool_raw: pd.DataFrame | None, *, random_state: int) -> pd.DataFrame:
    if train_pool_raw is None:
        return imbalanced_raw.iloc[0:0]
    used_index = set(imbalanced_raw.index)
    extra = train_pool_raw.loc[
        train_pool_raw["label"].eq("minority") & ~train_pool_raw.index.isin(used_index)
    ]
    return extra.sample(frac=1.0, random_state=random_state)


def _direct_generate(
    X: np.ndarray,
    minority_pos: np.ndarray,
    n_samples: int,
    *,
    method: str,
    n_neighbors: int,
    lambda_range: tuple[float, float],
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    minority = X[minority_pos]
    if len(minority) < 2:
        raise ValueError("At least two minority rows are required for synthetic generation")
    generated = []
    for _ in range(n_samples):
        anchor = int(rng.integers(0, len(minority)))
        candidates = _nearest_positions(minority, anchor, n_neighbors)
        neigh = int(rng.choice(candidates))
        lam = float(rng.uniform(*lambda_range))
        if method == "smote":
            generated.append((1.0 - lam) * minority[anchor] + lam * minority[neigh])
        else:
            to_pos = int(rng.choice(_nearest_positions(minority, neigh, n_neighbors)))
            generated.append(minority[anchor] + lam * (minority[to_pos] - minority[neigh]))
    return np.asarray(generated, dtype=float)


def _nearest_positions(X: np.ndarray, pos: int, n_neighbors: int) -> np.ndarray:
    distances = np.linalg.norm(X - X[pos], axis=1)
    order = np.argsort(distances)
    order = order[order != pos]
    if len(order) == 0:
        return np.asarray([pos])
    return order[: max(1, min(n_neighbors, len(order)))]


def _latent_generate(
    method: str,
    *,
    imbalanced_raw: pd.DataFrame,
    prepared: PreparedData,
    dataset_key: str,
    n_samples: int,
    random_state: int,
    n_neighbors: int,
    lambda_range: tuple[float, float],
    mimic_mode: str,
    mimic_capacity: float,
) -> np.ndarray:
    roles = feature_roles(imbalanced_raw, dataset_key)
    policy = GenerationPolicy(
        method="smote" if method == "latent_smote" else "displacement",
        neighbour_mode="normal",
        n_neighbors=n_neighbors,
        lambda_range=lambda_range,
    )
    mimic_columns = {
        "ignore": roles["ignore"],
        "regression": roles["numeric"],
        "classification": roles["categorical"] + ["label"],
    }
    model = MIMIC(
        columns=mimic_columns,
        mode=mimic_mode,
        capacity=mimic_capacity,
        policy=policy,
        bootstrap=False,
        random_state=random_state,
        feature_n_jobs=1,
    ).fit(imbalanced_raw)
    generated = model.sample(n_samples, condition={"label": "minority"}).drop(columns=["label"], errors="ignore")
    return transform_frame(
        prepared.preprocessor,
        generated,
        numeric_columns=prepared.numeric_columns,
        categorical_columns=prepared.categorical_columns,
    )
