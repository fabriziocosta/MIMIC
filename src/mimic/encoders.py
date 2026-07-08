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
