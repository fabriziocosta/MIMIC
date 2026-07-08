"""Iterated MIMIC estimator."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from .encoders import IdentityEncoder
from .mimic import MIMIC


DEFAULT_BASE_LEVEL = {"mode": "direct", "capacity": 0.2, "bootstrap": False}
DEFAULT_HIGHER_LEVEL = {"mode": "direct", "capacity": 0.1, "bootstrap": False}


class IteratedMIMIC(BaseEstimator, TransformerMixin):
    """Stack MIMIC models over successive learned representation tables."""

    def __init__(
        self,
        n_steps: int = 2,
        base_level: dict | None = None,
        higher_level: dict | None = None,
        levels: list[dict] | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_steps = n_steps
        self.base_level = base_level
        self.higher_level = higher_level
        self.levels = levels
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        del y
        X0 = pd.DataFrame(X).copy()
        self.input_columns_ = list(X0.columns)
        self.input_dtypes_ = X0.dtypes.to_dict()
        self.level_specs_ = self._resolve_level_specs()
        self.models_ = []
        self.representations_ = [X0]
        current = X0
        current_block_slices = None

        for level_index, spec in enumerate(self.level_specs_):
            fit_spec = self._fit_spec(
                spec,
                level_index=level_index,
                current=current,
                block_slices=current_block_slices,
            )
            if self.verbose:
                print(f"Fitting IteratedMIMIC level {level_index}: {fit_spec}")
            if level_index == 0:
                model = MIMIC(**fit_spec).fit(current)
            else:
                model = VectorBlockLevel(**fit_spec).fit(current)
            embedding = model.transform(current)
            current = self._embedding_frame(embedding)
            current_block_slices = model.embedding_slices_
            self.models_.append(model)
            self.representations_.append(current)

        self.representation_shapes_ = [frame.shape for frame in self.representations_]
        self.top_embeddings_ = self.representations_[-1].to_numpy(dtype=float)
        self.persistence_metadata_ = self._persistence_metadata()
        return self

    def transform(self, X):
        check_is_fitted(self, "models_")
        current = pd.DataFrame(X).copy()
        for model in self.models_:
            current = self._embedding_frame(model.transform(current))
        return current.to_numpy(dtype=float)

    def transform_levels(self, X):
        check_is_fitted(self, "models_")
        levels = [pd.DataFrame(X).copy()]
        current = levels[0]
        for model in self.models_:
            current = self._embedding_frame(model.transform(current))
            levels.append(current)
        return levels

    def inverse_transform(self, Z):
        check_is_fitted(self, "models_")
        return self.decode_embedding(Z)

    def decode_embedding(self, H, from_level: int | None = None):
        check_is_fitted(self, "models_")
        level = len(self.models_) if from_level is None else self._normalize_level(from_level)
        current = self._embedding_frame(H)
        for model in reversed(self.models_[:level]):
            current = model.decode(current.to_numpy(dtype=float))
        return current

    def decode_levels(self, Z, from_level: int | None = None):
        check_is_fitted(self, "models_")
        level = len(self.models_) if from_level is None else self._normalize_level(from_level)
        decoded = []
        current = self._embedding_frame(Z)
        for model in reversed(self.models_[:level]):
            current = model.decode(current.to_numpy(dtype=float))
            decoded.append(current)
        return decoded

    def sample(self, n_samples: int, level="top", return_trace: bool = False):
        check_is_fitted(self, "models_")
        level_index = self._sample_level_index(level)
        sampled = self.models_[level_index - 1].sample(n_samples, return_trace=return_trace)
        if return_trace:
            sampled_frame, trace = sampled
        else:
            sampled_frame, trace = sampled, None
        decoded = self.decode_embedding(sampled_frame.to_numpy(dtype=float), from_level=level_index - 1)
        if return_trace:
            return decoded, trace
        return decoded

    def save(self, path):
        check_is_fitted(self, "models_")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.persistence_metadata_ = self._persistence_metadata()
        joblib.dump(self, output_path)
        return self

    @classmethod
    def load(cls, path):
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise ValueError("Loaded artifact is not an IteratedMIMIC model")
        if not hasattr(model, "models_"):
            raise ValueError("Loaded IteratedMIMIC model is not fitted")
        return model

    def _resolve_level_specs(self):
        if self.levels is not None:
            specs = [deepcopy(spec) for spec in self.levels]
            if not specs:
                raise ValueError("levels must contain at least one level specification")
            return specs

        try:
            n_steps = int(self.n_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("n_steps must be a positive integer") from exc
        if n_steps < 1:
            raise ValueError("n_steps must be a positive integer")

        base = deepcopy(DEFAULT_BASE_LEVEL)
        if self.base_level is not None:
            base.update(deepcopy(self.base_level))
        higher = deepcopy(DEFAULT_HIGHER_LEVEL)
        if self.higher_level is not None:
            higher.update(deepcopy(self.higher_level))
        return [base] + [deepcopy(higher) for _ in range(n_steps - 1)]

    def _fit_spec(self, spec: dict, *, level_index: int, current: pd.DataFrame, block_slices):
        fit_spec = deepcopy(spec)
        if "random_state" not in fit_spec and self.random_state is not None:
            fit_spec["random_state"] = int(self.random_state) + level_index
        if "verbose" not in fit_spec:
            fit_spec["verbose"] = self.verbose
        if level_index > 0:
            fit_spec["block_slices"] = block_slices
            fit_spec["columns"] = list(current.columns)
        return fit_spec

    def _sample_level_index(self, level):
        if level == "top":
            return len(self.models_)
        return self._normalize_level(level)

    def _normalize_level(self, level):
        try:
            value = int(level)
        except (TypeError, ValueError) as exc:
            raise ValueError("level must be 'top' or an integer level from 0 to n_steps") from exc
        if not 0 <= value <= len(self.models_):
            raise ValueError(f"level must be between 0 and {len(self.models_)}")
        return value

    @staticmethod
    def _embedding_frame(embedding):
        arr = np.asarray(embedding, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        columns = [f"z{i:04d}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=columns)

    def _persistence_metadata(self):
        try:
            package_version = version("mimic")
        except PackageNotFoundError:
            package_version = None
        return {
            "package": "mimic",
            "package_version": package_version,
            "python_version": sys.version.split()[0],
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "estimator": "IteratedMIMIC",
            "n_steps": len(getattr(self, "models_", [])),
            "level_specs": deepcopy(getattr(self, "level_specs_", [])),
            "representation_shapes": list(getattr(self, "representation_shapes_", [])),
            "input_columns": list(getattr(self, "input_columns_", [])),
        }


class VectorBlockLevel(BaseEstimator, TransformerMixin):
    """MIMIC-like vector feature level for iterated embeddings.

    Each previous-level feature is represented by one embedding block. This
    level predicts each whole block from the other blocks and uses the encoder
    representation as the next-level block embedding.
    """

    def __init__(
        self,
        block_slices,
        columns=None,
        mode="direct",
        capacity: float = 0.1,
        encoder=None,
        decoder=None,
        bootstrap: bool = False,
        random_state: int | None = None,
        n_jobs: int | None = None,
        feature_n_jobs: int | None = 1,
        verbose: bool = False,
        **kwargs,
    ):
        self.block_slices = block_slices
        self.columns = columns
        self.mode = mode
        self.capacity = capacity
        self.encoder = encoder
        self.decoder = decoder
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.feature_n_jobs = feature_n_jobs
        self.verbose = verbose
        self.kwargs = kwargs

    def fit(self, X, y=None):
        del y
        X_arr = np.asarray(pd.DataFrame(X), dtype=float)
        self.input_columns_ = list(pd.DataFrame(X).columns)
        self.block_slices_ = {name: slice(sl.start, sl.stop) for name, sl in self.block_slices.items()}
        self.block_models_ = {}
        self._embedding_slices_ = {}
        parts = []
        start = 0
        for block_index, (name, sl) in enumerate(self.block_slices_.items()):
            context_indices = self._context_indices(X_arr.shape[1], sl)
            X_context = X_arr[:, context_indices]
            Y_block = X_arr[:, sl]
            encoder = self._new_encoder(block_index)
            encoder.fit(X_context, Y_block)
            H_block = self._to_2d(encoder.transform(X_context))
            decoder = self._new_decoder()
            decoder.fit(H_block, Y_block)
            self.block_models_[name] = {
                "encoder": encoder,
                "decoder": decoder,
                "context_indices": context_indices,
                "input_slice": sl,
                "embedding_dim": H_block.shape[1],
            }
            self._embedding_slices_[name] = slice(start, start + H_block.shape[1])
            start += H_block.shape[1]
            parts.append(H_block)
        self.embedding_slices_ = self._embedding_slices_
        self.train_X_ = pd.DataFrame(X).copy()
        self.train_embeddings_ = np.hstack(parts) if parts else np.empty((len(X_arr), 0))
        return self

    def transform(self, X):
        check_is_fitted(self, "block_models_")
        X_arr = np.asarray(pd.DataFrame(X), dtype=float)
        parts = []
        self._embedding_slices_ = {}
        start = 0
        for name, block in self.block_models_.items():
            H_block = self._to_2d(block["encoder"].transform(X_arr[:, block["context_indices"]]))
            parts.append(H_block)
            self._embedding_slices_[name] = slice(start, start + H_block.shape[1])
            start += H_block.shape[1]
        self.embedding_slices_ = self._embedding_slices_
        return np.hstack(parts) if parts else np.empty((len(X_arr), 0))

    def decode(self, H):
        check_is_fitted(self, "block_models_")
        H_arr = self._to_2d(H)
        expected_width = sum(sl.stop - sl.start for sl in self.embedding_slices_.values())
        if H_arr.shape[1] != expected_width:
            raise ValueError(f"H must have {expected_width} columns, got {H_arr.shape[1]}")
        blocks = []
        for name, block in self.block_models_.items():
            emb_sl = self.embedding_slices_[name]
            predicted = np.asarray(block["decoder"].predict(H_arr[:, emb_sl]), dtype=float)
            if predicted.ndim == 1:
                predicted = predicted.reshape(-1, 1)
            blocks.append(predicted)
        data = np.hstack(blocks) if blocks else np.empty((len(H_arr), 0))
        return pd.DataFrame(data, columns=self.input_columns_)

    def sample(self, n_samples: int, return_trace: bool = False):
        check_is_fitted(self, "train_embeddings_")
        rng = np.random.default_rng(self.random_state)
        H = np.asarray(self.train_embeddings_, dtype=float)
        if len(H) < 2:
            raise ValueError("At least two rows are required to sample")
        samples = []
        traces = []
        for sample_index in range(int(n_samples)):
            anchor = int(rng.integers(0, len(H)))
            neighbor = int(rng.integers(0, len(H) - 1))
            if neighbor >= anchor:
                neighbor += 1
            lam = float(rng.uniform(0.25, 0.75))
            samples.append((1.0 - lam) * H[anchor] + lam * H[neighbor])
            traces.append(
                {
                    "sample_index": sample_index,
                    "method": "smote",
                    "anchor_index": anchor,
                    "neighbour_index": neighbor,
                    "lambda": lam,
                }
            )
        decoded = self.decode(np.vstack(samples))
        if return_trace:
            return decoded, pd.DataFrame(traces)
        return decoded

    def _new_encoder(self, block_index: int):
        if self.encoder is not None:
            return deepcopy(self.encoder)
        mode = MIMIC._normalize_mode(self.mode, argument_name="mode")
        if mode == "identity":
            return IdentityEncoder(task="regression")
        params = MIMIC._capacity_parameters(MIMIC._validate_capacity(self.capacity))
        return MIMIC(
            capacity=self.capacity,
            random_state=None if self.random_state is None else int(self.random_state) + block_index,
        )._capacity_resnet_encoder(params)

    def _new_decoder(self):
        if self.decoder is not None:
            return deepcopy(self.decoder)
        return LinearRegression()

    @staticmethod
    def _context_indices(width: int, excluded: slice):
        mask = np.ones(width, dtype=bool)
        mask[excluded] = False
        return np.flatnonzero(mask)

    @staticmethod
    def _to_2d(X):
        arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr
