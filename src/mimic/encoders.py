"""Encoder implementations for MIMIC."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm


@dataclass
class SharedFeatureGroup:
    """Configuration for numerical columns trained by one shared ResNet."""

    columns: list[str]
    encoder: object
    coordinates: np.ndarray | None = None
    target_chunk_size: int = 64


class IdentityEncoder(BaseEstimator, TransformerMixin):
    """Pass-through encoder for original/preprocessed-space generation baselines.

    MIMIC treats this encoder as a special generation baseline and gives it the
    full modelled row rather than the usual target-excluded context.
    """

    include_target_context = True

    def __init__(self, task: str = "regression", sparse: bool = False):
        self.task = task
        self.sparse = sparse

    def fit(self, X, y=None):
        if self.task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        if self.sparse:
            return sparse.csr_matrix(X)
        return X.toarray() if sparse.issparse(X) else np.asarray(X)


class RandomForestPathEncoder(BaseEstimator, TransformerMixin):
    """Random-forest encoder using path or leaf sparse representations."""

    def __init__(
        self,
        task: str = "regression",
        embedding_dim: int | None = None,
        n_estimators: int = 100,
        max_depth: int | None = None,
        embedding: str = "auto",
        svd_random_state: int | None = None,
        sparse: bool = True,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ):
        self.task = task
        self.embedding_dim = embedding_dim
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.embedding = embedding
        self.svd_random_state = svd_random_state
        self.sparse = sparse
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if self.task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")
        if self.embedding not in {"auto", "path", "leaf"}:
            raise ValueError("embedding must be 'auto', 'path', or 'leaf'")

        if self.task == "classification":
            self.forest_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            self.forest_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        self.forest_.fit(X, y)
        self.embedding_ = self._resolve_embedding()
        Z = self._native_transform(X, fit=True)
        self.native_dim_ = Z.shape[1]
        if self.embedding_dim is not None:
            n_components = min(int(self.embedding_dim), max(1, Z.shape[1] - 1), Z.shape[0])
            self.svd_ = TruncatedSVD(
                n_components=n_components,
                random_state=self.svd_random_state,
            )
            self.svd_.fit(Z)
            self.output_dim_ = n_components
        else:
            self.svd_ = None
            self.output_dim_ = Z.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "forest_")
        Z = self._native_transform(X, fit=False)
        if self.svd_ is not None:
            return self.svd_.transform(Z)
        if self.sparse:
            return Z.tocsr()
        return Z.toarray()

    def _resolve_embedding(self) -> str:
        if self.embedding != "auto":
            return self.embedding
        return "path" if self.task == "classification" else "leaf"

    def _native_transform(self, X, fit: bool):
        if self.embedding_ == "path":
            return self.forest_.decision_path(X)[0].tocsr()

        leaves = self.forest_.apply(X)
        if fit:
            self.leaf_encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            return self.leaf_encoder_.fit_transform(leaves).tocsr()
        return self.leaf_encoder_.transform(leaves).tocsr()


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - import availability is environment dependent
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


if nn is not None:

    class _ResidualBlock(nn.Module):
        def __init__(self, dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x) + x


    class _TabularResNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, n_layers: int, dropout: float, out_dim: int):
            super().__init__()
            blocks = [
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            for _ in range(max(1, math.ceil(n_layers / 2))):
                blocks.append(_ResidualBlock(hidden_dim, dropout))
            blocks.extend(
                [
                    nn.Linear(hidden_dim, embedding_dim),
                    nn.BatchNorm1d(embedding_dim),
                    nn.ReLU(),
                ]
            )
            self.encoder = nn.Sequential(*blocks)
            self.head = nn.Linear(embedding_dim, out_dim)

        def forward(self, x):
            emb = self.encoder(x)
            return self.head(emb)

        def transform(self, x):
            return self.encoder(x)


    class _FiLMResidualBlock(nn.Module):
        def __init__(self, dim: int, descriptor_dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.film = nn.Linear(descriptor_dim, 2 * dim)

        def forward(self, x, descriptor):
            gamma, beta = self.film(descriptor).chunk(2, dim=1)
            return x + self.net(x) * (1.0 + gamma) + beta


    class _SharedFiLMResNet(nn.Module):
        def __init__(
            self,
            input_dim: int,
            n_targets: int,
            coordinate_descriptors: np.ndarray,
            feature_embedding_dim: int,
            hidden_dim: int,
            embedding_dim: int,
            n_layers: int,
            dropout: float,
        ):
            super().__init__()
            self.feature_embedding = nn.Embedding(n_targets, feature_embedding_dim)
            coordinate_tensor = torch.as_tensor(coordinate_descriptors, dtype=torch.float32)
            self.register_buffer("coordinate_descriptors", coordinate_tensor)
            descriptor_dim = feature_embedding_dim + coordinate_tensor.shape[1]
            self.input = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.blocks = nn.ModuleList(
                _FiLMResidualBlock(hidden_dim, descriptor_dim, dropout)
                for _ in range(max(1, math.ceil(n_layers / 2)))
            )
            self.embedding = nn.Sequential(
                nn.Linear(hidden_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(),
            )
            self.head = nn.Linear(embedding_dim, 1)

        def descriptor(self, target_indices):
            learned = self.feature_embedding(target_indices)
            coordinates = self.coordinate_descriptors[target_indices]
            return torch.cat([learned, coordinates], dim=1)

        def transform(self, x, target_indices):
            descriptor = self.descriptor(target_indices)
            hidden = self.input(x)
            for block in self.blocks:
                hidden = block(hidden, descriptor)
            return self.embedding(hidden)

        def forward(self, x, target_indices):
            return self.head(self.transform(x, target_indices)).reshape(-1)


class ResNetEncoder(BaseEstimator, TransformerMixin):
    """Small PyTorch tabular ResNet encoder."""

    def __init__(
        self,
        task: str = "regression",
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 10,
        validation_fraction: float = 0.1,
        restore_best_checkpoint: bool = True,
        random_state: int | None = None,
        device: str = "auto",
    ):
        self.task = task
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.restore_best_checkpoint = restore_best_checkpoint
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        if torch is None:
            raise ImportError("ResNetEncoder requires torch")
        if self.task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")

        X_arr = self._as_float32_array(X)
        rng = np.random.default_rng(self.random_state)
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))

        if self.task == "classification":
            self.classes_, y_arr = np.unique(y, return_inverse=True)
            y_tensor = torch.as_tensor(y_arr, dtype=torch.long)
            out_dim = len(self.classes_)
            criterion = nn.CrossEntropyLoss()
        else:
            y_arr = np.asarray(y, dtype=np.float32)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            y_tensor = torch.as_tensor(y_arr, dtype=torch.float32)
            out_dim = y_arr.shape[1]
            criterion = nn.MSELoss()

        indices = np.arange(len(X_arr))
        rng.shuffle(indices)
        n_val = int(round(len(indices) * self.validation_fraction))
        if n_val > 0 and len(indices) - n_val >= 2:
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
        else:
            val_idx = np.array([], dtype=int)
            train_idx = indices

        self.device_ = self._resolve_device()
        self.model_ = _TabularResNet(
            input_dim=X_arr.shape[1],
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            out_dim=out_dim,
        ).to(self.device_)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        train_ds = TensorDataset(
            torch.as_tensor(X_arr[train_idx], dtype=torch.float32),
            y_tensor[train_idx],
        )
        loader = DataLoader(train_ds, batch_size=self._safe_batch_size(len(train_ds)), shuffle=True)
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0

        for _epoch in range(self.max_epochs):
            self.model_.train()
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad()
                pred = self.model_(xb)
                loss = criterion(pred, yb if self.task == "classification" else yb)
                loss.backward()
                optimizer.step()

            val_loss = self._loss(X_arr[val_idx], y_tensor[val_idx], criterion) if len(val_idx) else self._loss(X_arr[train_idx], y_tensor[train_idx], criterion)
            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_state = deepcopy(self.model_.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if self.restore_best_checkpoint and best_state is not None:
            self.model_.load_state_dict(best_state)
        self.output_dim_ = self.embedding_dim
        return self

    def transform(self, X):
        check_is_fitted(self, "model_")
        X_arr = self._as_float32_array(X)
        self.model_.eval()
        with torch.no_grad():
            Xt = torch.as_tensor(X_arr, dtype=torch.float32, device=self.device_)
            return self.model_.transform(Xt).cpu().numpy()

    def _loss(self, X, y_tensor, criterion) -> float:
        self.model_.eval()
        with torch.no_grad():
            xb = torch.as_tensor(X, dtype=torch.float32, device=self.device_)
            yb = y_tensor.to(self.device_)
            pred = self.model_(xb)
            loss = criterion(pred, yb if self.task == "classification" else yb)
        return float(loss.detach().cpu().item())

    def _resolve_device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _safe_batch_size(self, n_samples: int) -> int:
        batch_size = min(int(self.batch_size), int(n_samples))
        if n_samples > 1 and batch_size > 1 and n_samples % batch_size == 1:
            batch_size -= 1
        return max(1, batch_size)

    def _as_float32_array(self, X):
        if sparse.issparse(X):
            return X.toarray().astype(np.float32, copy=False)
        return np.asarray(X, dtype=np.float32)


class SharedResNetEncoder(BaseEstimator):
    """Residual MLP shared across numerical targets and conditioned with FiLM."""

    def __init__(
        self,
        embedding_dim: int = 64,
        feature_embedding_dim: int = 8,
        coordinate_frequencies: int = 4,
        hidden_dim: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 10,
        validation_fraction: float = 0.1,
        restore_best_checkpoint: bool = True,
        random_state: int | None = None,
        device: str = "auto",
    ):
        self.embedding_dim = embedding_dim
        self.feature_embedding_dim = feature_embedding_dim
        self.coordinate_frequencies = coordinate_frequencies
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.restore_best_checkpoint = restore_best_checkpoint
        self.random_state = random_state
        self.device = device

    def fit(
        self,
        X,
        y,
        *,
        value_indices,
        missing_indices,
        coordinates=None,
        target_chunk_size: int = 64,
        row_ids=None,
        show_progress: bool = False,
    ):
        if torch is None:
            raise ImportError("SharedResNetEncoder requires torch")
        X_arr = self._as_float32_array(X)
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim != 2 or y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("y must have shape (n_rows, n_targets)")
        self.n_targets_ = y_arr.shape[1]
        self.value_indices_ = np.asarray(value_indices, dtype=int)
        self.missing_indices_ = np.asarray(missing_indices, dtype=int)
        if self.value_indices_.shape != (self.n_targets_,) or self.missing_indices_.shape != (self.n_targets_,):
            raise ValueError("value_indices and missing_indices must contain one index per target")
        self.coordinates_ = self._validate_coordinates(coordinates, self.n_targets_)
        self.coordinate_descriptors_ = self.fourier_descriptors(
            self.coordinates_, frequencies=self.coordinate_frequencies
        )
        self.target_chunk_size_ = self._validate_target_chunk_size(target_chunk_size)
        self.device_ = self._resolve_device()
        rng = np.random.default_rng(self.random_state)
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))

        row_ids = np.arange(len(X_arr)) if row_ids is None else np.asarray(row_ids)
        if row_ids.shape != (len(X_arr),):
            raise ValueError("row_ids must contain one identifier per input row")
        unique_row_ids = np.unique(row_ids)
        rng.shuffle(unique_row_ids)
        n_val = int(round(len(unique_row_ids) * self.validation_fraction))
        if n_val > 0 and len(unique_row_ids) - n_val >= 2:
            validation_ids = unique_row_ids[:n_val]
            validation_mask = np.isin(row_ids, validation_ids)
            self.validation_rows_ = np.flatnonzero(validation_mask)
            self.training_rows_ = np.flatnonzero(~validation_mask)
        else:
            self.validation_rows_ = np.array([], dtype=int)
            self.training_rows_ = np.arange(len(X_arr))
        self.training_row_ids_ = np.unique(row_ids[self.training_rows_])
        self.validation_row_ids_ = np.unique(row_ids[self.validation_rows_])

        self.model_ = _SharedFiLMResNet(
            input_dim=X_arr.shape[1],
            n_targets=self.n_targets_,
            coordinate_descriptors=self.coordinate_descriptors_,
            feature_embedding_dim=self.feature_embedding_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device_)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        epochs = tqdm(
            range(self.max_epochs),
            desc="Shared ResNet epochs",
            leave=False,
            disable=not show_progress,
        )
        for _epoch in epochs:
            self.model_.train()
            shuffled_rows = rng.permutation(self.training_rows_)
            for targets in self._target_chunks():
                rows_per_batch = max(1, int(self.batch_size) // len(targets))
                for row_start in range(0, len(shuffled_rows), rows_per_batch):
                    rows = shuffled_rows[row_start : row_start + rows_per_batch]
                    pair = self._make_pairs(X_arr, y_arr, rows, targets)
                    if pair is None:
                        continue
                    xb, target_ids, yb = pair
                    if len(yb) < 2:
                        continue
                    optimizer.zero_grad()
                    prediction = self.model_(xb, target_ids)
                    loss = nn.functional.mse_loss(prediction, yb)
                    loss.backward()
                    optimizer.step()
            validation_rows = self.validation_rows_ if len(self.validation_rows_) else self.training_rows_
            validation_loss = self._loss(X_arr, y_arr, validation_rows)
            epochs.set_postfix(validation_loss=f"{validation_loss:.5f}")
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_state = deepcopy(self.model_.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if self.restore_best_checkpoint and best_state is not None:
            self.model_.load_state_dict(best_state)
        epochs.close()
        self.output_dim_ = int(self.embedding_dim)
        return self

    def transform_target(self, X, target_index: int):
        check_is_fitted(self, "model_")
        target_index = int(target_index)
        if not 0 <= target_index < self.n_targets_:
            raise IndexError("target_index is outside the fitted target range")
        X_arr = self._as_float32_array(X)
        rows = np.arange(len(X_arr))
        masked = X_arr.copy()
        masked[:, self.value_indices_[target_index]] = 0.0
        masked[:, self.missing_indices_[target_index]] = 1.0
        self.model_.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(masked), max(1, int(self.batch_size))):
                xb = torch.as_tensor(masked[start : start + max(1, int(self.batch_size))], dtype=torch.float32, device=self.device_)
                target_ids = torch.full((len(xb),), target_index, dtype=torch.long, device=self.device_)
                outputs.append(self.model_.transform(xb, target_ids).cpu().numpy())
        return np.vstack(outputs) if outputs else np.empty((0, self.output_dim_), dtype=np.float32)

    @staticmethod
    def fourier_descriptors(coordinates, frequencies: int = 4):
        coordinates = np.asarray(coordinates, dtype=np.float32)
        if coordinates.ndim != 2:
            raise ValueError("coordinates must be a 2D array")
        if frequencies < 0:
            raise ValueError("coordinate_frequencies must be non-negative")
        parts = [coordinates]
        for frequency in range(int(frequencies)):
            scale = (2**frequency) * math.pi
            parts.extend([np.sin(scale * coordinates), np.cos(scale * coordinates)])
        return np.hstack(parts).astype(np.float32, copy=False) if parts else np.empty((len(coordinates), 0), dtype=np.float32)

    def _make_pairs(self, X, y, rows, targets):
        X_block = np.repeat(X[rows], len(targets), axis=0)
        target_ids_np = np.tile(targets, len(rows))
        y_block = y[np.repeat(rows, len(targets)), target_ids_np]
        observed = np.isfinite(y_block)
        if not np.any(observed):
            return None
        X_block = X_block[observed]
        target_ids_np = target_ids_np[observed]
        y_block = y_block[observed]
        pair_rows = np.arange(len(target_ids_np))
        X_block[pair_rows, self.value_indices_[target_ids_np]] = 0.0
        X_block[pair_rows, self.missing_indices_[target_ids_np]] = 1.0
        return (
            torch.as_tensor(X_block, dtype=torch.float32, device=self.device_),
            torch.as_tensor(target_ids_np, dtype=torch.long, device=self.device_),
            torch.as_tensor(y_block, dtype=torch.float32, device=self.device_),
        )

    def _loss(self, X, y, rows):
        self.model_.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for targets in self._target_chunks():
                rows_per_batch = max(1, int(self.batch_size) // len(targets))
                for row_start in range(0, len(rows), rows_per_batch):
                    row_block = rows[row_start : row_start + rows_per_batch]
                    pair = self._make_pairs(X, y, row_block, targets)
                    if pair is None:
                        continue
                    xb, target_ids, yb = pair
                    loss = nn.functional.mse_loss(self.model_(xb, target_ids), yb, reduction="sum")
                    total += float(loss.cpu().item())
                    count += len(yb)
        return total / max(1, count)

    def _target_chunks(self):
        for start in range(0, self.n_targets_, self.target_chunk_size_):
            yield np.arange(start, min(start + self.target_chunk_size_, self.n_targets_), dtype=int)

    @staticmethod
    def _validate_coordinates(coordinates, n_targets):
        if coordinates is None:
            return np.empty((n_targets, 0), dtype=np.float32)
        array = np.asarray(coordinates, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != n_targets:
            raise ValueError("coordinates must have one row per target column")
        if not np.isfinite(array).all():
            raise ValueError("coordinates must contain only finite values")
        return array

    @staticmethod
    def _validate_target_chunk_size(value):
        if not isinstance(value, (int, np.integer)) or int(value) <= 0:
            raise ValueError("target_chunk_size must be a positive integer")
        return int(value)

    def _resolve_device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @staticmethod
    def _as_float32_array(X):
        if sparse.issparse(X):
            return X.toarray().astype(np.float32, copy=False)
        return np.asarray(X, dtype=np.float32)


class SharedTargetEncoder:
    """Target-specific view of a fitted shared encoder."""

    def __init__(self, shared_encoder: SharedResNetEncoder, target_index: int):
        self.shared_encoder = shared_encoder
        self.target_index = int(target_index)
        self.output_dim_ = shared_encoder.output_dim_

    def transform(self, X):
        return self.shared_encoder.transform_target(X, self.target_index)
