"""Main MIMIC estimator."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
import warnings

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

from .decoders import IdentityDecoder, MixedFeatureDecoder, NeuralConditionalSampler
from .encoders import (
    IdentityEncoder,
    RandomForestPathEncoder,
    ResNetEncoder,
    SharedFeatureGroup,
    SharedResNetEncoder,
    SharedTargetEncoder,
)
from .policies import GenerationPolicy


@dataclass
class ContextPreprocessor:
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    include_missing_indicators: bool = True

    def fit(self, X: pd.DataFrame):
        X = pd.DataFrame(X)
        self.numeric_columns_ = [c for c in self.numeric_columns if c in X.columns]
        self.categorical_columns_ = [c for c in self.categorical_columns if c in X.columns]

        if self.numeric_columns_:
            self.numeric_imputer_ = SimpleImputer(strategy="median")
            self.scaler_ = StandardScaler()
            X_num = self.numeric_imputer_.fit_transform(X[self.numeric_columns_])
            self.scaler_.fit(X_num)
        else:
            self.numeric_imputer_ = None
            self.scaler_ = None

        if self.categorical_columns_:
            self.categorical_imputer_ = SimpleImputer(strategy="constant", fill_value="__missing__")
            self.onehot_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            X_cat = self.categorical_imputer_.fit_transform(X[self.categorical_columns_].astype("object"))
            self.onehot_.fit(X_cat)
        else:
            self.categorical_imputer_ = None
            self.onehot_ = None
        return self

    def transform(self, X: pd.DataFrame):
        X = pd.DataFrame(X)
        parts = []
        if self.numeric_columns_:
            X_num = self.numeric_imputer_.transform(X[self.numeric_columns_])
            X_num = self.scaler_.transform(X_num)
            parts.append(sparse.csr_matrix(X_num))
        if self.categorical_columns_:
            X_cat = self.categorical_imputer_.transform(X[self.categorical_columns_].astype("object"))
            parts.append(self.onehot_.transform(X_cat).tocsr())
        if self.include_missing_indicators and (self.numeric_columns_ or self.categorical_columns_):
            missing = X[self.numeric_columns_ + self.categorical_columns_].isna().astype(float).to_numpy()
            parts.append(sparse.csr_matrix(missing))
        if not parts:
            return sparse.csr_matrix((len(X), 0))
        return sparse.hstack(parts, format="csr")


@dataclass
class GlobalContextPreprocessor:
    include_missing_indicators: bool = True

    def fit(self, X: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]):
        X = pd.DataFrame(X)
        self.numeric_columns_ = [c for c in numeric_columns if c in X.columns]
        self.categorical_columns_ = [c for c in categorical_columns if c in X.columns]
        self.model_columns_ = self.numeric_columns_ + self.categorical_columns_

        if self.numeric_columns_:
            self.numeric_imputer_ = SimpleImputer(strategy="median")
            self.scaler_ = StandardScaler()
            X_num = self.numeric_imputer_.fit_transform(X[self.numeric_columns_])
            self.scaler_.fit(X_num)
        else:
            self.numeric_imputer_ = None
            self.scaler_ = None

        if self.categorical_columns_:
            self.categorical_imputer_ = SimpleImputer(strategy="constant", fill_value="__missing__")
            self.onehot_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            X_cat = self.categorical_imputer_.fit_transform(X[self.categorical_columns_].astype("object"))
            self.onehot_.fit(X_cat)
        else:
            self.categorical_imputer_ = None
            self.onehot_ = None

        self._build_column_indices()
        return self

    def transform_all(self, X: pd.DataFrame):
        X = pd.DataFrame(X)
        parts = []
        if self.numeric_columns_:
            X_num = self.numeric_imputer_.transform(X[self.numeric_columns_])
            X_num = self.scaler_.transform(X_num)
            parts.append(sparse.csr_matrix(X_num))
        if self.categorical_columns_:
            X_cat = self.categorical_imputer_.transform(X[self.categorical_columns_].astype("object"))
            parts.append(self.onehot_.transform(X_cat).tocsr())
        if self.include_missing_indicators and self.model_columns_:
            missing = X[self.model_columns_].isna().astype(float).to_numpy()
            parts.append(sparse.csr_matrix(missing))
        if not parts:
            return sparse.csr_matrix((len(X), 0))
        return sparse.hstack(parts, format="csr")

    def context_matrix(self, Xp_all, context_columns: list[str]):
        return Xp_all[:, self.encoded_indices_for(context_columns)]

    def encoded_indices_for(self, columns: list[str]):
        indices = []
        for column in columns:
            indices.extend(self.column_indices_.get(column, []))
        return np.asarray(sorted(indices), dtype=int)

    def encoded_indices_without(self, excluded_columns: list[str]):
        excluded = set(excluded_columns)
        columns = [column for column in self.model_columns_ if column not in excluded]
        return self.encoded_indices_for(columns)

    def _build_column_indices(self):
        self.column_indices_ = {column: [] for column in self.model_columns_}
        self.numeric_value_indices_ = {}
        self.categorical_onehot_indices_ = {}
        self.missing_indicator_indices_ = {}

        offset = 0
        for column in self.numeric_columns_:
            idx = np.asarray([offset], dtype=int)
            self.numeric_value_indices_[column] = idx
            self.column_indices_[column].extend(idx.tolist())
            offset += 1

        for column, categories in zip(self.categorical_columns_, getattr(self.onehot_, "categories_", [])):
            width = len(categories)
            idx = np.arange(offset, offset + width, dtype=int)
            self.categorical_onehot_indices_[column] = idx
            self.column_indices_[column].extend(idx.tolist())
            offset += width

        if self.include_missing_indicators:
            for column in self.model_columns_:
                idx = np.asarray([offset], dtype=int)
                self.missing_indicator_indices_[column] = idx
                self.column_indices_[column].extend(idx.tolist())
                offset += 1

        self.column_indices_ = {
            column: np.asarray(indices, dtype=int)
            for column, indices in self.column_indices_.items()
        }
        self.output_dim_ = offset


@dataclass
class BootstrapMember:
    encoder: object
    decoder: MixedFeatureDecoder
    embedding_dim: int


@dataclass
class FeatureModule:
    target_column: str
    task: str
    context_columns: list[str]
    context_indices: np.ndarray
    full_member: BootstrapMember
    members: list[BootstrapMember]
    observed_mask: pd.Series
    label_encoder: LabelEncoder | None = None
    target_scaler: StandardScaler | None = None
    bias: float | None = None
    noise: float | None = None
    classes_: np.ndarray | None = None


@dataclass
class SharedGroupModule:
    name: str
    columns: list[str]
    full_encoder: SharedResNetEncoder
    encoders: list[SharedResNetEncoder]


@dataclass
class NearestNeighborPrivacyFilter:
    enabled: bool = True
    k: int = 5
    min_ambiguous_neighbors: int = 3
    distance_ratio: float = 1.25
    exclude_generation_sources: bool = True
    max_attempt_multiplier: int = 10
    batch_multiplier: int = 3


class MIMIC(BaseEstimator, TransformerMixin):
    """Modular feature-wise estimator for imputation, confidence, and generation."""

    def __init__(
        self,
        columns="auto",
        encoder=None,
        decoder=None,
        policy=None,
        generation_decode_mode: str = "auto",
        mode="factorised",
        level=None,
        capacity: float = 0.25,
        n_bootstrap: int | None = None,
        bootstrap: bool = True,
        random_state: int | None = None,
        n_jobs: int | None = None,
        feature_n_jobs: int | None = 1,
        verbose: bool = False,
        classification_calibration: str = "none",
        regression_calibration: str = "none",
        calibration_interval_levels: tuple[float, ...] = (0.8, 0.9, 0.95),
        shared_feature_groups: dict[str, SharedFeatureGroup] | None = None,
        show_progress: bool = True,
    ):
        self.columns = columns
        self.encoder = encoder
        self.decoder = decoder
        self.policy = policy
        self.generation_decode_mode = generation_decode_mode
        self.mode = mode
        self.level = level
        self.capacity = capacity
        self.n_bootstrap = n_bootstrap
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.feature_n_jobs = feature_n_jobs
        self.verbose = verbose
        self.classification_calibration = classification_calibration
        self.regression_calibration = regression_calibration
        self.calibration_interval_levels = calibration_interval_levels
        self.shared_feature_groups = shared_feature_groups
        self.show_progress = show_progress
        if self.verbose:
            self._verbose_init()

    def fit(self, X, y=None):
        X = self._as_dataframe(X).copy()
        self._validate_schema(X)
        self._resolve_mode_configuration()
        self._validate_shared_feature_groups()
        if not isinstance(self.show_progress, bool):
            raise ValueError("show_progress must be a bool")
        self._verbose_fit_configuration()
        rng = np.random.default_rng(self.random_state)
        self.train_X_ = X.copy()
        self.train_index_ = X.index.copy()
        self.input_dtypes_ = X.dtypes.to_dict()
        self.feature_modules_ = {}
        self.classification_calibrators_ = {}
        self.regression_calibrators_ = {}
        self.calibration_scores_ = []
        feature_specs = []
        for column in self.model_columns_:
            task = self._task_for(column)
            observed_mask = X[column].notna()
            observed_indices = np.flatnonzero(observed_mask.to_numpy())
            if len(observed_indices) < 2:
                raise ValueError(f"Column {column!r} has too few observed rows")
            if task == "classification" and X.loc[observed_mask, column].nunique() < 2:
                raise ValueError(f"Classification column {column!r} needs at least two observed classes")
            if column in self.shared_columns_:
                continue
            if self.bootstrap_:
                bootstrap_samples = [
                    rng.choice(np.arange(len(observed_indices)), size=len(observed_indices), replace=True)
                    for _ in range(self.n_bootstrap_)
                ]
            else:
                bootstrap_samples = []
            feature_specs.append((column, bootstrap_samples))

        self.global_preprocessor_ = GlobalContextPreprocessor(include_missing_indicators=True)
        self.global_preprocessor_.fit(
            X,
            numeric_columns=self.regression_columns_,
            categorical_columns=self.classification_columns_,
        )
        Xp_all = self.global_preprocessor_.transform_all(X)

        fit_progress = tqdm(
            total=len(feature_specs) + len(self.shared_feature_groups_) + 4,
            desc="MIMIC fit",
            disable=not self.show_progress,
        )

        if self.feature_n_jobs in (None, 1):
            fitted_features = []
            for column, bootstrap_samples in feature_specs:
                fit_progress.set_postfix(stage=f"feature {column}")
                fitted_features.append(
                    self._fit_feature_module(
                        X, Xp_all, column=column, bootstrap_samples=bootstrap_samples
                    )
                )
                fit_progress.update(1)
        else:
            fit_progress.set_postfix(stage="parallel features")
            fitted_features = Parallel(n_jobs=self.feature_n_jobs)(
                delayed(self._fit_feature_module)(X, Xp_all, column=column, bootstrap_samples=bootstrap_samples)
                for column, bootstrap_samples in feature_specs
            )
            fit_progress.update(len(feature_specs))
        feature_modules = {column: module for column, module, _data in fitted_features}
        oob_calibration_data = {column: data for column, _module, data in fitted_features}
        self.shared_group_modules_ = {}
        for group_name, group in self.shared_feature_groups_.items():
            fit_progress.set_postfix(stage=f"shared group {group_name}")
            group_modules, group_module, group_oob = self._fit_shared_group(
                X, Xp_all, group_name=group_name, group=group, rng=rng
            )
            feature_modules.update(group_modules)
            self.shared_group_modules_[group_name] = group_module
            oob_calibration_data.update(group_oob)
            fit_progress.update(1)
        self.feature_modules_ = {
            column: feature_modules[column] for column in self.model_columns_
        }

        fit_progress.set_postfix(stage="training embeddings")
        self.train_embeddings_ = self.transform(X)
        fit_progress.update(1)
        self.embedding_slices_ = self._embedding_slices_
        fit_progress.set_postfix(stage="calibration")
        self._fit_calibrators(oob_calibration_data)
        fit_progress.update(1)
        fit_progress.set_postfix(stage="conditional decoders")
        self._fit_conditional_samplers()
        if self.generation_decode_mode_config_ == "joint":
            fit_progress.set_postfix(stage="joint decoder")
            self._fit_joint_decoder()
        fit_progress.update(1)
        self.generation_decode_mode_ = self._resolve_generation_decode_mode()
        self.policy_ = self._new_policy()
        self.policy_.validate()
        fit_progress.set_postfix(stage="neighbours")
        self.neighbour_index_ = self._fit_neighbours(self.train_embeddings_)
        fit_progress.update(1)
        fit_progress.set_postfix(stage="complete")
        fit_progress.close()
        self._verbose_fit_summary()
        return self

    def _fit_shared_group(self, X, Xp_all, *, group_name, group, rng):
        columns = list(group.columns)
        n_rows = len(X)
        n_targets = len(columns)
        target_scalers = {}
        y_scaled = np.full((n_rows, n_targets), np.nan, dtype=np.float32)
        for target_index, column in enumerate(columns):
            observed = X[column].notna().to_numpy()
            scaler = StandardScaler()
            values = X.loc[observed, column].astype(float).to_numpy().reshape(-1, 1)
            scaler.fit(values)
            y_scaled[observed, target_index] = scaler.transform(values).ravel()
            target_scalers[column] = scaler

        value_indices = [
            int(self.global_preprocessor_.numeric_value_indices_[column][0]) for column in columns
        ]
        missing_indices = [
            int(self.global_preprocessor_.missing_indicator_indices_[column][0]) for column in columns
        ]
        full_encoder = self._fit_shared_encoder(
            group,
            Xp_all,
            y_scaled,
            value_indices=value_indices,
            missing_indices=missing_indices,
            encoder_index=0,
            row_ids=np.arange(n_rows),
        )
        full_decoders = {}
        for target_index, column in enumerate(columns):
            observed_rows = np.flatnonzero(np.isfinite(y_scaled[:, target_index]))
            embedding = full_encoder.transform_target(Xp_all[observed_rows], target_index)
            decoder = self._new_decoder()
            decoder.fit_target(column, "regression", embedding, y_scaled[observed_rows, target_index])
            full_decoders[column] = decoder

        bootstrap_encoders = []
        bootstrap_decoders = {column: [] for column in columns}
        oob_errors = {column: [] for column in columns}
        oob_predictions = {column: [] for column in columns}
        oob_true = {column: [] for column in columns}
        if self.bootstrap_:
            for bootstrap_index in range(self.n_bootstrap_):
                sampled_rows = self._shared_bootstrap_rows(y_scaled, rng)
                encoder = self._fit_shared_encoder(
                    group,
                    Xp_all[sampled_rows],
                    y_scaled[sampled_rows],
                    value_indices=value_indices,
                    missing_indices=missing_indices,
                    encoder_index=bootstrap_index + 1,
                    row_ids=sampled_rows,
                )
                bootstrap_encoders.append(encoder)
                oob_rows = np.setdiff1d(np.arange(n_rows), np.unique(sampled_rows))
                for target_index, column in enumerate(columns):
                    observed_sample = np.isfinite(y_scaled[sampled_rows, target_index])
                    train_rows = sampled_rows[observed_sample]
                    embedding = encoder.transform_target(Xp_all[train_rows], target_index)
                    decoder = self._new_decoder()
                    decoder.fit_target(column, "regression", embedding, y_scaled[train_rows, target_index])
                    bootstrap_decoders[column].append(decoder)
                    observed_oob = oob_rows[np.isfinite(y_scaled[oob_rows, target_index])]
                    if len(observed_oob):
                        oob_embedding = encoder.transform_target(Xp_all[observed_oob], target_index)
                        prediction_scaled = decoder.predict_target(column, oob_embedding).astype(float)
                        scaler = target_scalers[column]
                        prediction = self._inverse_regression_target(prediction_scaled, scaler)
                        true = X.iloc[observed_oob][column].astype(float).to_numpy()
                        oob_errors[column].extend((prediction - true).tolist())
                        oob_predictions[column].extend(prediction.tolist())
                        oob_true[column].extend(true.tolist())

        all_context_indices = np.arange(self.global_preprocessor_.output_dim_, dtype=int)
        modules = {}
        calibration = {}
        for target_index, column in enumerate(columns):
            full_member = BootstrapMember(
                encoder=SharedTargetEncoder(full_encoder, target_index),
                decoder=full_decoders[column],
                embedding_dim=full_encoder.output_dim_,
            )
            members = [
                BootstrapMember(
                    encoder=SharedTargetEncoder(encoder, target_index),
                    decoder=decoder,
                    embedding_dim=encoder.output_dim_,
                )
                for encoder, decoder in zip(bootstrap_encoders, bootstrap_decoders[column])
            ]
            errors = oob_errors[column]
            modules[column] = FeatureModule(
                target_column=column,
                task="regression",
                context_columns=list(self.model_columns_),
                context_indices=all_context_indices,
                full_member=full_member,
                members=members,
                observed_mask=X[column].notna(),
                target_scaler=target_scalers[column],
                bias=float(np.mean(errors)) if errors else 0.0,
                noise=float(np.var(errors, ddof=1)) if len(errors) > 1 else 0.0,
            )
            calibration[column] = {
                "task": "regression",
                "predictions": np.asarray(oob_predictions[column], dtype=float),
                "true": np.asarray(oob_true[column], dtype=float),
            }
        return (
            modules,
            SharedGroupModule(
                name=group_name,
                columns=columns,
                full_encoder=full_encoder,
                encoders=bootstrap_encoders,
            ),
            calibration,
        )

    @staticmethod
    def _shared_bootstrap_rows(y_scaled, rng, max_attempts: int = 100):
        n_rows = len(y_scaled)
        for _attempt in range(max_attempts):
            sampled_rows = rng.choice(np.arange(n_rows), size=n_rows, replace=True)
            observed_counts = np.isfinite(y_scaled[sampled_rows]).sum(axis=0)
            if np.all(observed_counts >= 2):
                return sampled_rows
        raise ValueError(
            "Could not draw a row-level shared-group bootstrap with two observed values per target"
        )

    def _fit_shared_encoder(
        self,
        group,
        X,
        y,
        *,
        value_indices,
        missing_indices,
        encoder_index,
        row_ids,
    ):
        encoder = clone(group.encoder)
        if self.random_state is not None and "random_state" in encoder.get_params():
            encoder.set_params(random_state=int(self.random_state + encoder_index))
        return encoder.fit(
            X,
            y,
            value_indices=value_indices,
            missing_indices=missing_indices,
            coordinates=group.coordinates,
            target_chunk_size=group.target_chunk_size,
            row_ids=row_ids,
            show_progress=self.show_progress,
        )

    def _fit_feature_module(self, X: pd.DataFrame, Xp_all, *, column: str, bootstrap_samples: list[np.ndarray]):
        task = self._task_for(column)
        observed_mask = X[column].notna()
        observed_indices = np.flatnonzero(observed_mask.to_numpy())

        context_columns = self._context_columns_for(column)
        context_indices = self.global_preprocessor_.encoded_indices_for(context_columns)
        label_encoder = None
        y_observed = X.loc[observed_mask, column]
        if task == "classification":
            label_encoder = LabelEncoder()
            y_full = label_encoder.fit_transform(y_observed.astype(str))
            classes = label_encoder.classes_
            target_scaler = None
        else:
            target_scaler = StandardScaler()
            y_full = target_scaler.fit_transform(y_observed.astype(float).to_numpy().reshape(-1, 1)).ravel()
            classes = None

        members = []
        oob_errors = []
        oob_regression_predictions = []
        oob_regression_true = []
        oob_class_probabilities = []
        oob_class_true = []
        full_member = self._fit_feature_member(
            Xp_all,
            column=column,
            task=task,
            rows=observed_indices,
            y=y_full,
            context_indices=context_indices,
            encoder_index=0,
        )
        for b, sample_pos in enumerate(bootstrap_samples):
            sample_rows = observed_indices[sample_pos]
            sampled_observed_positions = sample_pos
            sampled_y = y_full[sampled_observed_positions]

            member = self._fit_feature_member(
                Xp_all,
                column=column,
                task=task,
                rows=sample_rows,
                y=sampled_y,
                context_indices=context_indices,
                encoder_index=b + 1,
            )
            members.append(member)

            oob_source = np.setdiff1d(np.arange(len(observed_indices)), np.unique(sampled_observed_positions))
            if len(oob_source):
                oob_rows = observed_indices[oob_source]
                Xp_oob = Xp_all[oob_rows][:, context_indices]
                H_oob = self._to_2d(member.encoder.transform(Xp_oob), width=member.embedding_dim)
                pred = member.decoder.predict_target(column, H_oob)
                if task == "regression":
                    pred_raw = self._inverse_regression_target(pred.astype(float), target_scaler)
                    true_raw = self._inverse_regression_target(y_full[oob_source].astype(float), target_scaler)
                    err = pred_raw - true_raw
                    oob_errors.extend(err.tolist())
                    oob_regression_predictions.extend(pred_raw.tolist())
                    oob_regression_true.extend(true_raw.tolist())
                else:
                    p = self._align_proba_to_classes(
                        member.decoder.predict_proba_target(column, H_oob),
                        member.decoder.models_[column],
                        len(classes),
                    )
                    oob_class_probabilities.append(p)
                    oob_class_true.extend(y_full[oob_source].astype(int).tolist())

        bias = float(np.mean(oob_errors)) if oob_errors else 0.0
        noise = float(np.var(oob_errors, ddof=1)) if len(oob_errors) > 1 else 0.0
        module = FeatureModule(
            target_column=column,
            task=task,
            context_columns=context_columns,
            context_indices=context_indices,
            full_member=full_member,
            members=members,
            observed_mask=observed_mask,
            label_encoder=label_encoder,
            target_scaler=target_scaler,
            bias=bias,
            noise=noise,
            classes_=classes,
        )
        if task == "regression":
            oob_calibration_data = {
                "task": task,
                "predictions": np.asarray(oob_regression_predictions, dtype=float),
                "true": np.asarray(oob_regression_true, dtype=float),
            }
        else:
            oob_calibration_data = {
                "task": task,
                "probabilities": np.vstack(oob_class_probabilities) if oob_class_probabilities else np.empty((0, len(classes))),
                "true": np.asarray(oob_class_true, dtype=int),
            }
        return column, module, oob_calibration_data

    def _fit_feature_member(
        self,
        Xp_all,
        *,
        column: str,
        task: str,
        rows: np.ndarray,
        y: np.ndarray,
        context_indices: np.ndarray,
        encoder_index: int,
    ):
        Xp = Xp_all[rows][:, context_indices]
        encoder = self._new_encoder(task, encoder_index)
        encoder.fit(Xp, y)
        H = self._to_2d(encoder.transform(Xp))
        decoder = self._new_decoder()
        decoder.fit_target(column, task, H, y)
        return BootstrapMember(
            encoder=encoder,
            decoder=decoder,
            embedding_dim=H.shape[1],
        )

    def transform(self, X):
        check_is_fitted(self, "feature_modules_")
        X = self._as_dataframe(X)
        Xp_all = self.global_preprocessor_.transform_all(X)
        parts = []
        self._embedding_slices_ = {}
        start = 0
        for column, module in self.feature_modules_.items():
            Xp = Xp_all[:, module.context_indices]
            block = self._to_2d(
                module.full_member.encoder.transform(Xp),
                width=module.full_member.embedding_dim,
            )
            parts.append(block)
            self._embedding_slices_[column] = slice(start, start + block.shape[1])
            start += block.shape[1]
        return np.hstack(parts) if parts else np.empty((len(X), 0))

    def decode(self, H):
        """Decode embedding rows back into the fitted feature space."""

        check_is_fitted(self, "feature_modules_")
        H = self._to_2d(H)
        expected_width = sum(sl.stop - sl.start for sl in self.embedding_slices_.values())
        if H.shape[1] != expected_width:
            raise ValueError(f"H must have {expected_width} columns, got {H.shape[1]}")
        return self._decode_embeddings(H)

    def impute(self, X, columns=None, return_confidence: bool = False):
        check_is_fitted(self, "feature_modules_")
        X = self._as_dataframe(X).copy()
        target_columns = self._selected_columns(columns, only_missing=True, X=X)
        for column in target_columns:
            pred, _details = self._predict_column(X, column)
            mask = X[column].isna()
            if self.feature_modules_[column].task == "regression":
                X.loc[mask, column] = np.asarray(pred, dtype=float)[mask.to_numpy()]
            else:
                X[column] = X[column].astype(object)
                X.loc[mask, column] = np.asarray(pred, dtype=object)[mask.to_numpy()]
        if return_confidence:
            return X, self.confidence(X, columns=target_columns)
        return X

    def confidence(self, X, columns=None):
        check_is_fitted(self, "feature_modules_")
        X = self._as_dataframe(X)
        rows = []
        for column in self._selected_columns(columns, only_missing=False, X=X):
            pred, details = self._predict_column(X, column)
            module = self.feature_modules_[column]
            for i, idx in enumerate(X.index):
                observed = X.iloc[i][column] if column in X.columns else np.nan
                base = {
                    "row_index": idx,
                    "column": column,
                    "task": module.task,
                    "prediction": pred[i],
                    "observed": observed,
                    "bias": module.bias if module.task == "regression" else np.nan,
                    "variance": details.get("variance", [np.nan] * len(X))[i],
                    "residual": self._residual(observed, pred[i], module.task),
                    "uncertainty": details.get("uncertainty", [np.nan] * len(X))[i],
                    "discrepancy": self._discrepancy(observed, pred[i], module.task),
                    "entropy": details.get("entropy", [np.nan] * len(X))[i],
                    "confidence": details.get("confidence", [np.nan] * len(X))[i],
                    "probabilities": details.get("probabilities", [None] * len(X))[i],
                    "predicted_probability": details.get("predicted_probability", [np.nan] * len(X))[i],
                    "probability_variance": details.get("probability_variance", [np.nan] * len(X))[i],
                    "probability_std": details.get("probability_std", [np.nan] * len(X))[i],
                    "probability_min": details.get("probability_min", [np.nan] * len(X))[i],
                    "probability_max": details.get("probability_max", [np.nan] * len(X))[i],
                    "probability_margin": details.get("probability_margin", [np.nan] * len(X))[i],
                    "vote_counts": details.get("vote_counts", [None] * len(X))[i],
                    "vote_fraction": details.get("vote_fraction", [np.nan] * len(X))[i],
                    "disagreement": details.get("disagreement", [np.nan] * len(X))[i],
                    "top_class": details.get("top_class", [None] * len(X))[i],
                    "second_class": details.get("second_class", [None] * len(X))[i],
                    "noise": module.noise if module.task == "regression" else np.nan,
                }
                intervals = details.get("intervals", {})
                for name, values in intervals.items():
                    base[name] = values[i]
                rows.append(base)
        return pd.DataFrame(rows)

    def calibration_report(self):
        check_is_fitted(self, "calibration_scores_")
        return pd.DataFrame(self.calibration_scores_)

    def sample(self, n_samples: int, condition=None, return_trace: bool = False, privacy_filter=None):
        check_is_fitted(self, "train_embeddings_")
        rng = np.random.default_rng(self.random_state)
        H = np.asarray(self.train_embeddings_)
        condition_mask = self._condition_mask(condition)
        anchor_candidates = np.flatnonzero(condition_mask)
        if len(anchor_candidates) == 0:
            raise ValueError("No training rows satisfy the requested generation condition")
        privacy_config = self._resolve_privacy_filter(privacy_filter)
        synth_embeddings, traces = self._sample_embeddings(
            n_samples,
            rng=rng,
            H=H,
            condition=condition,
            condition_mask=condition_mask,
            anchor_candidates=anchor_candidates,
            privacy_filter=privacy_config,
        )

        H_new = np.vstack(synth_embeddings)
        if self.generation_decode_mode_ == "joint":
            X_new = self._decode_joint_embeddings(H_new)
        else:
            X_new = self._decode_embeddings(H_new)
        if condition is not None:
            for column, value in condition.items():
                if column in X_new.columns:
                    X_new[column] = value
        if self.generation_decode_mode_ == "factorised":
            X_new, cell_traces = self._sample_factorised_decoded(
                H_new,
                X_new,
                condition=condition,
                rng=rng,
                n_passes=3,
                return_trace=return_trace,
            )
            traces.extend(cell_traces)
        X_new = self._restore_sample_schema(X_new)
        if return_trace:
            return X_new, pd.DataFrame(traces)
        return X_new

    def save(self, path):
        """Persist a fitted MIMIC model to a local joblib artifact."""
        try:
            check_is_fitted(self, "train_embeddings_")
        except Exception as exc:
            raise ValueError("Cannot save an unfitted MIMIC model") from exc

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.persistence_metadata_ = self._persistence_metadata()
        joblib.dump(self, output_path)
        return self

    @classmethod
    def load(cls, path):
        """Load a fitted MIMIC model from a local joblib artifact."""
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise ValueError("Loaded artifact is not a MIMIC model")
        if not hasattr(model, "train_embeddings_"):
            raise ValueError("Loaded MIMIC model is not fitted")
        return model

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
            "model_columns": list(getattr(self, "model_columns_", [])),
            "resolved_mode": getattr(self, "mode_", None),
            "resolved_generation_decode_mode": getattr(self, "generation_decode_mode_", None),
        }

    def _sample_embeddings(
        self,
        n_samples: int,
        *,
        rng,
        H,
        condition,
        condition_mask,
        anchor_candidates,
        privacy_filter,
    ):
        if privacy_filter is None or not privacy_filter.enabled:
            embeddings = []
            traces = []
            for sample_index in range(n_samples):
                h_new, trace, _sources = self._generate_embedding_candidate(
                    sample_index=sample_index,
                    generation_attempt=sample_index,
                    rng=rng,
                    H=H,
                    condition=condition,
                    condition_mask=condition_mask,
                    anchor_candidates=anchor_candidates,
                    privacy_filter_enabled=False,
                )
                embeddings.append(h_new)
                traces.append(trace)
            return embeddings, traces

        max_attempts = max(n_samples, int(np.ceil(n_samples * privacy_filter.max_attempt_multiplier)))
        embeddings = []
        traces = []
        attempt = 0
        while len(embeddings) < n_samples and attempt < max_attempts:
            remaining_attempts = max_attempts - attempt
            remaining_samples = n_samples - len(embeddings)
            batch_size = min(remaining_attempts, max(1, int(np.ceil(remaining_samples * privacy_filter.batch_multiplier))))
            for _ in range(batch_size):
                h_new, trace, sources = self._generate_embedding_candidate(
                    sample_index=len(embeddings),
                    generation_attempt=attempt,
                    rng=rng,
                    H=H,
                    condition=condition,
                    condition_mask=condition_mask,
                    anchor_candidates=anchor_candidates,
                    privacy_filter_enabled=True,
                )
                accepted, metrics = self._candidate_passes_privacy_filter(h_new, sources, privacy_filter)
                trace.update(metrics)
                trace["privacy_filter_accepted"] = bool(accepted)
                if accepted:
                    trace["sample_index"] = len(embeddings)
                    embeddings.append(h_new)
                    traces.append(trace)
                    if len(embeddings) == n_samples:
                        break
                attempt += 1
                if attempt >= max_attempts:
                    break

        if len(embeddings) < n_samples:
            raise ValueError(
                "Nearest-neighbor privacy filter accepted "
                f"{len(embeddings)} of {n_samples} requested samples after {attempt} attempts. "
                "Relax the filter or increase max_attempt_multiplier."
            )
        return embeddings, traces

    @staticmethod
    def _resolve_privacy_filter(privacy_filter):
        if privacy_filter is None or privacy_filter is False:
            return None
        if privacy_filter is True:
            privacy_filter = NearestNeighborPrivacyFilter()
        if not isinstance(privacy_filter, NearestNeighborPrivacyFilter):
            raise ValueError("privacy_filter must be None, a bool, or NearestNeighborPrivacyFilter")
        if privacy_filter.k < 1:
            raise ValueError("privacy_filter.k must be >= 1")
        if privacy_filter.min_ambiguous_neighbors < 1:
            raise ValueError("privacy_filter.min_ambiguous_neighbors must be >= 1")
        if privacy_filter.distance_ratio < 1:
            raise ValueError("privacy_filter.distance_ratio must be >= 1")
        if privacy_filter.max_attempt_multiplier < 1:
            raise ValueError("privacy_filter.max_attempt_multiplier must be >= 1")
        if privacy_filter.batch_multiplier < 1:
            raise ValueError("privacy_filter.batch_multiplier must be >= 1")
        return privacy_filter

    def _generate_embedding_candidate(
        self,
        *,
        sample_index: int,
        generation_attempt: int,
        rng,
        H,
        condition,
        condition_mask,
        anchor_candidates,
        privacy_filter_enabled: bool,
    ):
        anchor_pos = int(rng.choice(anchor_candidates))
        neigh_pos = self._choose_neighbour(anchor_pos, rng, condition_mask=condition_mask)
        lam = float(rng.uniform(*self.policy_.lambda_range))
        base_trace = {
            "trace_type": "embedding",
            "sample_index": sample_index,
            "lambda": lam,
            "neighbour_mode": self.policy_.neighbour_mode,
            "condition": json.dumps(condition, sort_keys=True) if condition is not None else None,
            "decoder": self._decoder_name(),
            "generation_decode_mode": self.generation_decode_mode,
            "resolved_generation_decode_mode": self.generation_decode_mode_,
            "random_state": self.random_state,
            "privacy_filter_enabled": bool(privacy_filter_enabled),
            "privacy_filter_accepted": True,
            "generation_attempt": generation_attempt,
        }
        if self.policy_.method == "smote":
            h_new = (1.0 - lam) * H[anchor_pos] + lam * H[neigh_pos]
            trace = {
                **base_trace,
                "method": "smote",
                "anchor_index": self.train_index_[anchor_pos],
                "neighbour_index": self.train_index_[neigh_pos],
            }
            sources = {anchor_pos, neigh_pos}
        else:
            from_pos = neigh_pos
            to_pos = self._choose_neighbour(from_pos, rng, condition_mask=condition_mask)
            h_new = H[anchor_pos] + lam * (H[to_pos] - H[from_pos])
            trace = {
                **base_trace,
                "method": "displacement",
                "anchor_index": self.train_index_[anchor_pos],
                "displacement_from_index": self.train_index_[from_pos],
                "displacement_to_index": self.train_index_[to_pos],
                "restriction": "basic",
            }
            sources = {anchor_pos, from_pos, to_pos}
        return h_new, trace, sources

    def _candidate_passes_privacy_filter(self, h_new, source_positions, privacy_filter: NearestNeighborPrivacyFilter):
        train_embeddings = np.asarray(self.train_embeddings_)
        n_train = len(train_embeddings)
        k = min(max(1, int(privacy_filter.k)), n_train)
        distances, indices = self._privacy_neighbour_distances(np.asarray(h_new, dtype=float).reshape(1, -1), k)
        distances = distances[0]
        indices = indices[0].astype(int)
        nearest_distance = float(distances[0])
        kth_distance = float(distances[-1])
        source_positions = set(int(pos) for pos in source_positions)
        generation_source_in_top_k = any(int(idx) in source_positions for idx in indices)
        positive_distances = distances[distances > np.finfo(float).eps]
        baseline_distance = nearest_distance if nearest_distance > np.finfo(float).eps else (
            float(positive_distances[0]) if len(positive_distances) else nearest_distance
        )
        threshold = baseline_distance * float(privacy_filter.distance_ratio)
        eligible = []
        for distance, idx in zip(distances, indices):
            if privacy_filter.exclude_generation_sources and int(idx) in source_positions:
                continue
            if float(distance) <= threshold:
                eligible.append(int(idx))
        ambiguous_count = len(eligible)
        accepted = ambiguous_count >= int(privacy_filter.min_ambiguous_neighbors)
        return accepted, {
            "nearest_distance": nearest_distance,
            "kth_distance": kth_distance,
            "ambiguous_neighbor_count": ambiguous_count,
            "generation_source_in_top_k": bool(generation_source_in_top_k),
        }

    def _privacy_neighbour_distances(self, H_query, k: int):
        model = NearestNeighbors(n_neighbors=k)
        model.fit(np.asarray(self.train_embeddings_))
        return model.kneighbors(H_query, return_distance=True)

    def plot(self, X=None, embedding_columns=None, color_by=None, center=None, random_state=None, ax=None):
        check_is_fitted(self, "feature_modules_")
        import matplotlib.pyplot as plt

        X = self.train_X_ if X is None else self._as_dataframe(X)
        original = self._plot_original_matrix(X)
        embedding = self._select_embedding_columns(self.transform(X), embedding_columns)
        orig_xy = self._classical_mds(original, center=center, random_state=random_state)
        emb_xy = self._classical_mds(embedding, center=center, random_state=random_state)

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        else:
            axes = np.asarray(ax).ravel()
            fig = axes[0].figure
            if len(axes) < 2:
                raise ValueError("ax must contain two axes")

        colors, is_numeric = self._plot_colors(X, color_by)
        for axis, xy, title in zip(axes[:2], [orig_xy, emb_xy], ["Original data MDS", "MIMIC embedding MDS"]):
            if colors is None:
                axis.scatter(xy[:, 0], xy[:, 1], s=24)
            elif is_numeric:
                sc = axis.scatter(xy[:, 0], xy[:, 1], c=colors, s=24, cmap="viridis")
                fig.colorbar(sc, ax=axis)
            else:
                codes, uniques = pd.factorize(colors.fillna("__missing__"))
                sc = axis.scatter(xy[:, 0], xy[:, 1], c=codes, s=24, cmap="tab10")
                axis.legend(
                    handles=sc.legend_elements()[0],
                    labels=[str(u) for u in uniques],
                    loc="best",
                    fontsize="small",
                )
            axis.set_title(title)
            axis.set_xlabel("MDS 1")
            axis.set_ylabel("MDS 2")
        fig.tight_layout()
        return fig, axes[:2]

    def _select_embedding_columns(self, embedding, embedding_columns):
        if embedding_columns is None:
            return embedding
        if isinstance(embedding_columns, str):
            embedding_columns = [embedding_columns]
        selected = list(embedding_columns)
        unknown = set(selected) - set(self.embedding_slices_)
        if unknown:
            raise ValueError(f"Unknown embedding columns: {sorted(unknown)}")
        if not selected:
            raise ValueError("embedding_columns must contain at least one column")
        indices = np.concatenate(
            [
                np.arange(self.embedding_slices_[column].start, self.embedding_slices_[column].stop)
                for column in selected
            ]
        )
        return embedding[:, indices]

    def _predict_column(self, X: pd.DataFrame, column: str):
        module = self.feature_modules_[column]
        Xp_all = self.global_preprocessor_.transform_all(X)
        prediction_members = self._prediction_members(module)
        if module.task == "regression":
            preds = []
            for member in prediction_members:
                H = self._member_embedding(member, module, X, Xp_all=Xp_all, max_width=member.embedding_dim)
                pred_scaled = member.decoder.predict_target(column, H).astype(float)
                preds.append(self._inverse_regression_target(pred_scaled, module.target_scaler))
            arr = np.vstack(preds)
            mean = arr.mean(axis=0)
            variance = arr.var(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
            details = {
                "variance": variance,
                "uncertainty": variance + (module.noise or 0.0),
            }
            if column in getattr(self, "regression_calibrators_", {}):
                intervals = {}
                for level, quantile in self.regression_calibrators_[column]["quantiles"].items():
                    suffix = self._interval_suffix(level)
                    intervals[f"lower_{suffix}"] = mean - quantile
                    intervals[f"upper_{suffix}"] = mean + quantile
                details["intervals"] = intervals
            return mean, details

        probas = []
        votes = []
        for member in prediction_members:
            H = self._member_embedding(member, module, X, Xp_all=Xp_all, max_width=member.embedding_dim)
            p = self._aligned_predict_proba(member, module, column, H)
            probas.append(p)
            votes.append(np.argmax(p, axis=1))
        p_arr = np.stack(probas, axis=0)
        p_mean = p_arr.mean(axis=0)
        p_mean = self._apply_classification_calibrator(column, p_mean)
        pred_codes = np.argmax(p_mean, axis=1)
        pred_labels = module.label_encoder.inverse_transform(pred_codes)
        pred_probs = p_mean[np.arange(len(X)), pred_codes]
        sorted_probs = np.sort(p_mean, axis=1)
        margin = sorted_probs[:, -1] - (sorted_probs[:, -2] if p_mean.shape[1] > 1 else 0.0)
        vote_arr = np.vstack(votes)
        vote_fraction = np.mean(vote_arr == pred_codes[None, :], axis=0)
        eps = 1e-12
        entropy = -np.sum(p_mean * np.log(p_mean + eps), axis=1)

        probabilities = []
        vote_counts = []
        second_class = []
        for i in range(len(X)):
            probabilities.append({str(cls): float(p_mean[i, k]) for k, cls in enumerate(module.classes_)})
            counts = np.bincount(vote_arr[:, i], minlength=len(module.classes_))
            vote_counts.append({str(module.classes_[k]): int(counts[k]) for k in range(len(module.classes_))})
            order = np.argsort(p_mean[i])
            second_class.append(module.classes_[order[-2]] if len(order) > 1 else None)

        class_probs = p_arr[:, np.arange(len(X)), pred_codes]
        prob_var = class_probs.var(axis=0, ddof=1) if p_arr.shape[0] > 1 else np.zeros(len(X))
        details = {
            "variance": prob_var,
            "uncertainty": entropy,
            "entropy": entropy,
            "confidence": pred_probs,
            "probabilities": probabilities,
            "predicted_probability": pred_probs,
            "probability_variance": prob_var,
            "probability_std": np.sqrt(prob_var),
            "probability_min": class_probs.min(axis=0),
            "probability_max": class_probs.max(axis=0),
            "probability_margin": margin,
            "vote_counts": vote_counts,
            "vote_fraction": vote_fraction,
            "disagreement": 1.0 - vote_fraction,
            "top_class": pred_labels,
            "second_class": second_class,
        }
        return pred_labels, details

    def _fit_calibrators(self, oob_calibration_data):
        self.classification_calibrators_ = {}
        self.regression_calibrators_ = {}
        self.calibration_scores_ = []
        for column, data in oob_calibration_data.items():
            if data["task"] == "classification":
                self._fit_classification_calibrator(column, data)
            else:
                self._fit_regression_calibrator(column, data)

    def _fit_classification_calibrator(self, column, data):
        method = self.classification_calibration
        if method == "none":
            return
        probabilities = np.asarray(data["probabilities"], dtype=float)
        y_true = np.asarray(data["true"], dtype=int)
        n_samples = len(y_true)
        module = self.feature_modules_[column]
        if n_samples < max(5, len(module.classes_)):
            self.calibration_scores_.append(
                {"column": column, "task": "classification", "method": method, "status": "skipped_insufficient_oob", "n_oob": n_samples}
            )
            return
        if method == "temperature":
            temperature = self._fit_temperature(probabilities, y_true)
            calibrated = self._temperature_calibrate(probabilities, temperature)
            self.classification_calibrators_[column] = {"method": method, "temperature": temperature}
        else:
            calibrators = []
            calibrated_columns = []
            for class_idx in range(probabilities.shape[1]):
                iso = IsotonicRegression(out_of_bounds="clip")
                y_binary = (y_true == class_idx).astype(float)
                if np.unique(y_binary).size < 2:
                    calibrators.append(None)
                    calibrated_columns.append(probabilities[:, class_idx])
                    continue
                iso.fit(probabilities[:, class_idx], y_binary)
                calibrators.append(iso)
                calibrated_columns.append(iso.predict(probabilities[:, class_idx]))
            calibrated = self._normalize_probabilities(np.vstack(calibrated_columns).T)
            self.classification_calibrators_[column] = {"method": method, "calibrators": calibrators}
        self.calibration_scores_.append(
            {
                "column": column,
                "task": "classification",
                "method": method,
                "status": "fitted",
                "n_oob": n_samples,
                "brier_before": self._multiclass_brier(probabilities, y_true),
                "brier_after": self._multiclass_brier(calibrated, y_true),
                "nll_before": self._multiclass_nll(probabilities, y_true),
                "nll_after": self._multiclass_nll(calibrated, y_true),
            }
        )

    def _fit_regression_calibrator(self, column, data):
        method = self.regression_calibration
        if method == "none":
            return
        predictions = np.asarray(data["predictions"], dtype=float)
        y_true = np.asarray(data["true"], dtype=float)
        n_samples = len(y_true)
        if n_samples < 2:
            self.calibration_scores_.append(
                {"column": column, "task": "regression", "method": method, "status": "skipped_insufficient_oob", "n_oob": n_samples}
            )
            return
        residuals = np.abs(predictions - y_true)
        quantiles = {}
        for level in self.calibration_interval_levels_:
            quantiles[float(level)] = float(np.quantile(residuals, float(level), method="higher"))
        self.regression_calibrators_[column] = {"method": method, "quantiles": quantiles}
        self.calibration_scores_.append(
            {
                "column": column,
                "task": "regression",
                "method": method,
                "status": "fitted",
                "n_oob": n_samples,
                "mean_absolute_oob_residual": float(np.mean(residuals)),
            }
        )

    def _apply_classification_calibrator(self, column, probabilities):
        calibrator = getattr(self, "classification_calibrators_", {}).get(column)
        if calibrator is None:
            return probabilities
        if calibrator["method"] == "temperature":
            return self._temperature_calibrate(probabilities, calibrator["temperature"])
        calibrated_columns = []
        for class_idx, iso in enumerate(calibrator["calibrators"]):
            if iso is None:
                calibrated_columns.append(probabilities[:, class_idx])
            else:
                calibrated_columns.append(iso.predict(probabilities[:, class_idx]))
        return self._normalize_probabilities(np.vstack(calibrated_columns).T)

    @staticmethod
    def _fit_temperature(probabilities, y_true):
        grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 40))
        scores = [MIMIC._multiclass_nll(MIMIC._temperature_calibrate(probabilities, t), y_true) for t in grid]
        return float(grid[int(np.argmin(scores))])

    @staticmethod
    def _temperature_calibrate(probabilities, temperature):
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
        logits = np.log(clipped)
        logits = logits / max(float(temperature), 1e-12)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return MIMIC._normalize_probabilities(exp_logits)

    @staticmethod
    def _normalize_probabilities(probabilities):
        arr = np.clip(np.asarray(probabilities, dtype=float), 0.0, np.inf)
        row_sums = arr.sum(axis=1, keepdims=True)
        missing = row_sums.ravel() <= 0
        if np.any(missing):
            arr[missing, :] = 1.0
            row_sums = arr.sum(axis=1, keepdims=True)
        return arr / row_sums

    @staticmethod
    def _multiclass_nll(probabilities, y_true):
        p = np.clip(probabilities[np.arange(len(y_true)), y_true], 1e-12, 1.0)
        return float(-np.mean(np.log(p)))

    @staticmethod
    def _multiclass_brier(probabilities, y_true):
        y = np.zeros_like(probabilities, dtype=float)
        y[np.arange(len(y_true)), y_true] = 1.0
        return float(np.mean(np.sum(np.square(probabilities - y), axis=1)))

    @staticmethod
    def _interval_suffix(level):
        return str(int(round(float(level) * 100)))

    def _member_embedding(
        self,
        member: BootstrapMember,
        module: FeatureModule,
        X: pd.DataFrame,
        Xp_all=None,
        max_width: int | None = None,
    ):
        if Xp_all is None:
            Xp_all = self.global_preprocessor_.transform_all(X)
        Xp = Xp_all[:, module.context_indices]
        return self._to_2d(member.encoder.transform(Xp), width=max_width)

    def _decode_embeddings(self, H):
        data = {}
        for column, module in self.feature_modules_.items():
            sl = self.embedding_slices_[column]
            block = H[:, sl]
            member = module.full_member
            if module.task == "regression":
                b = self._to_2d(block, width=member.embedding_dim)[:, : member.embedding_dim]
                pred_scaled = member.decoder.predict_target(column, b).astype(float)
                data[column] = self._inverse_regression_target(pred_scaled, module.target_scaler)
            else:
                b = self._to_2d(block, width=member.embedding_dim)[:, : member.embedding_dim]
                proba = self._aligned_predict_proba(member, module, column, b)
                pred_codes = np.argmax(proba, axis=1)
                data[column] = module.label_encoder.inverse_transform(pred_codes)
        return pd.DataFrame(data)

    def _fit_conditional_samplers(self):
        for column, module in self.feature_modules_.items():
            sl = self.embedding_slices_[column]
            observed_pos = np.flatnonzero(module.observed_mask.to_numpy())
            if len(observed_pos) == 0:
                continue
            H_context = self._without_slice(self.train_embeddings_[observed_pos], sl)
            y_observed = self.train_X_.iloc[observed_pos][column]
            if module.task == "classification":
                y = module.label_encoder.transform(y_observed.astype(str))
            else:
                y = module.target_scaler.transform(y_observed.astype(float).to_numpy().reshape(-1, 1)).ravel()
            train_indices = self.train_index_[observed_pos].to_numpy()
            member = module.full_member
            if hasattr(member.decoder, "fit_sampler_target"):
                member.decoder.fit_sampler_target(column, module.task, H_context, y, train_indices=train_indices)

    def _fit_joint_decoder(self):
        complete_mask = self.train_X_[self.model_columns_].notna().all(axis=1).to_numpy()
        complete_pos = np.flatnonzero(complete_mask)
        if len(complete_pos) < 2:
            raise ValueError("generation_decode_mode='joint' requires at least two complete modelled training rows")
        prototype = self._joint_decoder_prototype()
        evidence = self._joint_evidence(self.train_embeddings_[complete_pos])
        targets = {}
        target_specs = []
        for column, module in self.feature_modules_.items():
            values = self.train_X_.iloc[complete_pos][column]
            if module.task == "regression":
                targets[column] = module.target_scaler.transform(values.astype(float).to_numpy().reshape(-1, 1)).ravel()
                target_specs.append({"column": column, "task": "regression"})
            else:
                targets[column] = module.label_encoder.transform(values.astype(str))
                target_specs.append({"column": column, "task": "classification", "n_classes": len(module.classes_)})
        self.joint_decoder_ = clone(prototype)
        self.joint_decoder_.fit_joint_decoder(evidence, targets, target_specs)

    def _joint_decoder_prototype(self):
        prototype = None
        for module in self.feature_modules_.values():
            decoder = module.full_member.decoder
            if prototype is None:
                prototype = decoder
            if not hasattr(decoder, "conditional_evidence_target"):
                raise ValueError(
                    "generation_decode_mode='joint' requires NeuralConditionalSampler-style conditional evidence"
                )
        if prototype is None or not hasattr(prototype, "fit_joint_decoder"):
            raise ValueError("generation_decode_mode='joint' requires a decoder that can fit a joint row decoder")
        return prototype

    def _joint_evidence(self, H):
        blocks = []
        for column, module in self.feature_modules_.items():
            H_context = self._without_slice(H, self.embedding_slices_[column])
            member = module.full_member
            if not hasattr(member.decoder, "conditional_evidence_target"):
                raise ValueError(
                    "generation_decode_mode='joint' requires NeuralConditionalSampler-style conditional evidence"
                )
            blocks.append(member.decoder.conditional_evidence_target(column, module.task, H_context))
        return np.hstack(blocks)

    def _decode_joint_embeddings(self, H):
        check_is_fitted(self, "joint_decoder_")
        predictions = self.joint_decoder_.predict_joint(self._joint_evidence(H))
        data = {}
        for column, module in self.feature_modules_.items():
            pred = predictions[column]
            if module.task == "regression":
                data[column] = self._inverse_regression_target(pred, module.target_scaler)
            else:
                data[column] = module.label_encoder.inverse_transform(np.asarray(pred, dtype=int))
        return pd.DataFrame(data)

    def _has_stochastic_decoders(self):
        for column, module in self.feature_modules_.items():
            member = module.full_member
            if hasattr(member.decoder, "can_sample_target") and member.decoder.can_sample_target(column):
                return True
        return False

    def _sample_factorised_decoded(self, H, X, condition, rng, n_passes: int, return_trace: bool):
        X = X.copy()
        condition = condition or {}
        traces = []
        for sampling_pass in range(n_passes):
            for column, module in self.feature_modules_.items():
                if column in condition:
                    X[column] = condition[column]
                    continue
                member_index, member = self._sampleable_member(module, column, rng)
                if member is None:
                    continue
                H_context = self._without_slice(H, self.embedding_slices_[column])
                sampled, sample_traces = member.decoder.sample_target(
                    column,
                    module.task,
                    H_context,
                    rng,
                    return_trace=True,
                )
                if module.task == "regression":
                    values = self._inverse_regression_target(sampled, module.target_scaler)
                    X[column] = values
                else:
                    codes = np.asarray(sampled, dtype=int)
                    values = module.label_encoder.inverse_transform(codes)
                    X[column] = values
                if return_trace:
                    for row_pos, detail in enumerate(sample_traces):
                        trace = {
                            "trace_type": "cell",
                            "sample_index": row_pos,
                            "method": "factorised_conditional",
                            "sweep": sampling_pass,
                            "column": column,
                            "task": module.task,
                            "sampled_value": X.iloc[row_pos][column],
                            "conditioning": "z_minus_j",
                            "embedding_slice_start": self.embedding_slices_[column].start,
                            "embedding_slice_stop": self.embedding_slices_[column].stop,
                            "member_index": member_index,
                            "condition": json.dumps(condition, sort_keys=True) if condition else None,
                            "decoder": member.decoder.__class__.__name__,
                            "generation_decode_mode": self.generation_decode_mode,
                            "resolved_generation_decode_mode": self.generation_decode_mode_,
                        }
                        trace.update(detail)
                        traces.append(trace)
        for column, value in condition.items():
            if column in X.columns:
                X[column] = value
        return X, traces

    def _sampleable_member(self, module: FeatureModule, column: str, rng):
        member = module.full_member
        if not hasattr(member.decoder, "can_sample_target") or not member.decoder.can_sample_target(column):
            return None, None
        return -1, member

    @staticmethod
    def _prediction_members(module: FeatureModule):
        return module.members if module.members else [module.full_member]

    @staticmethod
    def _without_slice(H, sl):
        return np.hstack([H[:, : sl.start], H[:, sl.stop :]])

    def _aligned_predict_proba(self, member: BootstrapMember, module: FeatureModule, column: str, H):
        proba = member.decoder.predict_proba_target(column, H)
        model = member.decoder.models_[column]
        return self._align_proba_to_classes(proba, model, len(module.classes_))

    @staticmethod
    def _align_proba_to_classes(proba, model, n_classes: int):
        model_classes = getattr(model, "classes_", np.arange(proba.shape[1]))
        if len(model_classes) == n_classes and np.array_equal(model_classes, np.arange(n_classes)):
            return proba
        aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
        for local_idx, class_code in enumerate(model_classes):
            aligned[:, int(class_code)] = proba[:, local_idx]
        row_sums = aligned.sum(axis=1)
        missing = row_sums == 0
        if np.any(missing):
            aligned[missing, :] = 1.0 / n_classes
            row_sums = aligned.sum(axis=1)
        return aligned / row_sums[:, None]

    def _inverse_regression_target(self, values, target_scaler):
        arr = np.asarray(values, dtype=float).reshape(-1, 1)
        if target_scaler is None:
            return arr.ravel()
        return target_scaler.inverse_transform(arr).ravel()

    def _validate_schema(self, X: pd.DataFrame):
        self.columns_ = list(X.columns)
        if self.columns is None or self.columns == "auto":
            (
                self.ignore_columns_,
                self.regression_columns_,
                self.classification_columns_,
            ) = self._infer_columns(X)
        else:
            if not isinstance(self.columns, dict):
                raise ValueError(
                    "columns must be 'auto' or a mapping with 'ignore', 'regression', and 'classification' lists"
                )
            allowed = {"ignore", "regression", "classification"}
            unknown_keys = set(self.columns) - allowed
            if unknown_keys:
                raise ValueError("columns must only contain 'ignore', 'regression', and 'classification'")
            self.ignore_columns_ = list(self.columns.get("ignore", []))
            self.regression_columns_ = list(self.columns.get("regression", []))
            self.classification_columns_ = list(self.columns.get("classification", []))

        declared = set(self.ignore_columns_) | set(self.regression_columns_) | set(self.classification_columns_)
        missing = declared - set(X.columns)
        if missing:
            raise ValueError(f"Declared columns not present in X: {sorted(missing)}")
        overlap = (set(self.ignore_columns_) & set(self.regression_columns_)) | (
            set(self.ignore_columns_) & set(self.classification_columns_)
        ) | (set(self.regression_columns_) & set(self.classification_columns_))
        if overlap:
            raise ValueError(f"Columns cannot appear in multiple roles: {sorted(overlap)}")
        task_columns = set(self.regression_columns_) | set(self.classification_columns_)
        self.model_columns_ = [c for c in X.columns if c in task_columns]
        unassigned = set(X.columns) - declared
        if unassigned:
            raise ValueError(f"Every non-ignored column needs a task: {sorted(unassigned)}")
        if not self.model_columns_:
            raise ValueError("At least one modelled column is required")

    def _validate_shared_feature_groups(self):
        groups = {} if self.shared_feature_groups is None else self.shared_feature_groups
        if not isinstance(groups, dict):
            raise ValueError("shared_feature_groups must be a mapping of names to SharedFeatureGroup objects")
        shared_columns = set()
        validated = {}
        for name, group in groups.items():
            if not isinstance(name, str) or not name:
                raise ValueError("shared feature group names must be non-empty strings")
            if not isinstance(group, SharedFeatureGroup):
                raise ValueError(f"Shared group {name!r} must be a SharedFeatureGroup")
            columns = list(group.columns)
            if not columns:
                raise ValueError(f"Shared group {name!r} must contain at least one column")
            if len(columns) != len(set(columns)):
                raise ValueError(f"Shared group {name!r} contains duplicate columns")
            unknown = set(columns) - set(self.model_columns_)
            if unknown:
                raise ValueError(f"Shared group {name!r} contains unknown columns: {sorted(unknown)}")
            non_regression = set(columns) - set(self.regression_columns_)
            if non_regression:
                raise ValueError(
                    f"Shared group {name!r} only supports regression columns: {sorted(non_regression)}"
                )
            overlap = shared_columns & set(columns)
            if overlap:
                raise ValueError(f"Columns cannot appear in multiple shared groups: {sorted(overlap)}")
            if not isinstance(group.encoder, SharedResNetEncoder):
                raise ValueError(f"Shared group {name!r} requires a SharedResNetEncoder")
            coordinates = group.coordinates
            if coordinates is not None:
                array = np.asarray(coordinates)
                if array.ndim != 2 or array.shape[0] != len(columns):
                    raise ValueError(
                        f"Shared group {name!r} coordinates must have one row per column"
                    )
                if not np.isfinite(array.astype(float)).all():
                    raise ValueError(f"Shared group {name!r} coordinates must be finite")
            if not isinstance(group.target_chunk_size, (int, np.integer)) or int(group.target_chunk_size) <= 0:
                raise ValueError(f"Shared group {name!r} target_chunk_size must be positive")
            shared_columns.update(columns)
            validated[name] = group
        self.shared_feature_groups_ = validated
        self.shared_columns_ = shared_columns

    def _infer_columns(self, X: pd.DataFrame):
        ignore_columns = []
        regression_columns = []
        classification_columns = []
        for column in X.columns:
            series = X[column]
            non_null = series.dropna()
            if self._is_auto_ignore_column(column, non_null):
                ignore_columns.append(column)
            elif pd.api.types.is_numeric_dtype(series) and not self._is_small_integer_set(non_null):
                regression_columns.append(column)
            else:
                classification_columns.append(column)
        return ignore_columns, regression_columns, classification_columns

    def _is_auto_ignore_column(self, column, non_null: pd.Series):
        if self._is_id_like_name(column):
            return True
        n = len(non_null)
        if n < 2:
            return False
        return non_null.nunique(dropna=True) / n >= 0.95

    @staticmethod
    def _is_id_like_name(column):
        name = str(column).strip().lower()
        return (
            name == "id"
            or name.endswith("_id")
            or name.endswith("-id")
            or name.endswith(" id")
            or "identifier" in name
        )

    @staticmethod
    def _is_small_integer_set(non_null: pd.Series):
        if len(non_null) == 0:
            return False
        values = pd.to_numeric(non_null, errors="coerce")
        if values.isna().any():
            return False
        arr = values.to_numpy(dtype=float)
        if not np.all(np.isclose(arr, np.round(arr))):
            return False
        unique_count = len(pd.unique(arr))
        return 2 <= unique_count <= 10

    def _new_context_preprocessor(self, context_columns):
        numeric = [c for c in context_columns if c in self.regression_columns_]
        categorical = [c for c in context_columns if c in self.classification_columns_]
        return ContextPreprocessor(numeric_columns=numeric, categorical_columns=categorical)

    def _context_columns_for(self, column: str):
        encoder = getattr(self, "encoder_", self.encoder)
        if getattr(encoder, "include_target_context", False):
            return list(self.model_columns_)
        return [c for c in self.model_columns_ if c != column]

    def _restore_sample_schema(self, X: pd.DataFrame):
        X = X.copy()
        ordered = [c for c in self.model_columns_ if c in X.columns]
        X = X[ordered]
        for column in ordered:
            dtype = self.input_dtypes_.get(column)
            if dtype is None:
                continue
            if column in self.regression_columns_ and pd.api.types.is_integer_dtype(dtype):
                X[column] = np.rint(pd.to_numeric(X[column], errors="coerce")).astype(dtype)
            elif column in self.regression_columns_ and pd.api.types.is_bool_dtype(dtype):
                X[column] = X[column].astype(bool).astype(dtype)
            elif column in self.classification_columns_:
                try:
                    X[column] = X[column].astype(dtype)
                except (TypeError, ValueError):
                    pass
        return X

    def _new_encoder(self, task: str, bootstrap_index: int):
        encoder = self.encoder_
        encoder = clone(encoder)
        params = encoder.get_params()
        updates = {}
        if "task" in params:
            updates["task"] = task
        if "random_state" in params and self.random_state is not None:
            updates["random_state"] = int(self.random_state + bootstrap_index)
        if "n_jobs" in params and self.n_jobs is not None:
            updates["n_jobs"] = self.n_jobs
        if updates:
            encoder.set_params(**updates)
        return encoder

    def _new_decoder(self):
        return clone(self.decoder_)

    def _new_policy(self):
        return deepcopy(self.policy_config_)

    def _resolve_mode_configuration(self):
        self._validate_calibration_configuration()
        if not isinstance(self.bootstrap, bool):
            raise ValueError("bootstrap must be a bool")
        mode_name = self._resolve_mode_name()
        capacity = self._validate_capacity(self.capacity)
        preset_params = self._capacity_parameters(capacity)
        self.mode_ = mode_name
        self.level_ = self._mode_to_level(mode_name)
        self.capacity_ = capacity
        self.capacity_parameters_ = preset_params

        requested_bootstrap = preset_params["n_bootstrap"] if self.n_bootstrap is None else self.n_bootstrap
        self.bootstrap_ = bool(self.bootstrap)
        self.n_bootstrap_ = int(requested_bootstrap) if self.bootstrap_ else 0
        self.policy_config_ = self.policy if self.policy is not None else self._default_generation_policy()
        preset_encoder, preset_decoder, preset_mode = self._preset_components(mode_name, preset_params)

        self.encoder_ = self.encoder if self.encoder is not None else preset_encoder
        self.decoder_ = self.decoder if self.decoder is not None else preset_decoder
        use_preset_decoder = self.decoder is None
        self.generation_decode_mode_config_ = self.generation_decode_mode
        if self.generation_decode_mode == "auto" and mode_name is not None and use_preset_decoder:
            self.generation_decode_mode_config_ = preset_mode

    def _resolve_mode_name(self):
        mode_name = self._normalize_mode(self.mode, argument_name="mode")
        level_name = self._normalize_mode(self.level, argument_name="level") if self.level is not None else None
        if level_name is not None:
            if mode_name is not None and mode_name != "joint" and mode_name != level_name:
                raise ValueError("mode and level specify different presets")
            return level_name
        return mode_name

    @classmethod
    def _normalize_mode(cls, value, argument_name: str = "mode"):
        if value is None:
            return None
        numeric_modes = {
            0: "identity",
            1: "direct",
            2: "factorised",
            3: "joint",
        }
        string_modes = {
            "0": "identity",
            "1": "direct",
            "2": "factorised",
            "3": "joint",
            "identity": "identity",
            "direct": "direct",
            "deterministic": "direct",
            "factorised": "factorised",
            "factorized": "factorised",
            "probabilistic": "factorised",
            "joint": "joint",
        }
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            if int(value) in numeric_modes:
                return numeric_modes[int(value)]
        if isinstance(value, str):
            key = value.strip().lower().replace("-", "_")
            key = key.replace("_", "")
            if key in string_modes:
                return string_modes[key]
        raise ValueError(f"{argument_name} must be one of 0, 1, 2, 3, 'identity', 'direct', 'factorised', or 'joint'")

    @staticmethod
    def _mode_to_level(mode_name):
        if mode_name is None:
            return None
        return {
            "identity": 0,
            "direct": 1,
            "factorised": 2,
            "joint": 3,
        }[mode_name]

    @staticmethod
    def _validate_capacity(capacity):
        try:
            value = float(capacity)
        except (TypeError, ValueError) as exc:
            raise ValueError("capacity must be a number between 0 and 1") from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError("capacity must be between 0 and 1")
        return value

    def _validate_calibration_configuration(self):
        if self.classification_calibration not in {"none", "temperature", "isotonic"}:
            raise ValueError("classification_calibration must be one of 'none', 'temperature', or 'isotonic'")
        if self.regression_calibration not in {"none", "conformal"}:
            raise ValueError("regression_calibration must be one of 'none' or 'conformal'")
        try:
            levels = tuple(float(level) for level in self.calibration_interval_levels)
        except (TypeError, ValueError) as exc:
            raise ValueError("calibration_interval_levels must contain values between 0 and 1") from exc
        if not levels or any(level <= 0.0 or level >= 1.0 for level in levels):
            raise ValueError("calibration_interval_levels must contain values strictly between 0 and 1")
        self.calibration_interval_levels_ = levels

    @classmethod
    def _capacity_parameters(cls, capacity: float):
        return {
            "embedding_dim": cls._scale_int(capacity, 1, 128),
            "hidden_dim": cls._scale_int(capacity, 8, 128),
            "n_layers": cls._scale_int(capacity, 1, 8),
            "max_epochs": cls._scale_int(capacity, 10, 300),
            "patience": cls._scale_int(capacity, 2, 30),
            "batch_size": cls._scale_int(capacity, 32, 256),
            "n_components": cls._scale_int(capacity, 1, 8),
            "n_bootstrap": cls._scale_int(capacity, 1, 5),
            "dropout": cls._scale_float(capacity, 0.0, 0.2),
            "learning_rate": cls._scale_log(capacity, 3e-3, 3e-4),
            "weight_decay": cls._scale_log(capacity, 1e-6, 1e-3),
        }

    def _capacity_resnet_encoder(self, params):
        return ResNetEncoder(
            embedding_dim=params["embedding_dim"],
            hidden_dim=params["hidden_dim"],
            n_layers=params["n_layers"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            max_epochs=params["max_epochs"],
            patience=params["patience"],
            random_state=self.random_state,
        )

    def _capacity_neural_decoder(self, params):
        return NeuralConditionalSampler(
            n_components=params["n_components"],
            hidden_dim=params["hidden_dim"],
            n_layers=params["n_layers"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            max_epochs=params["max_epochs"],
            patience=params["patience"],
            random_state=self.random_state,
        )

    @staticmethod
    def _scale_int(capacity: float, low: int, high: int):
        return int(round(low + capacity * (high - low)))

    @staticmethod
    def _scale_float(capacity: float, low: float, high: float):
        return float(low + capacity * (high - low))

    @staticmethod
    def _scale_log(capacity: float, low: float, high: float):
        return float(np.exp(np.log(low) + capacity * (np.log(high) - np.log(low))))

    def _resolve_generation_decode_mode(self):
        valid_modes = {"auto", "direct", "factorised", "joint"}
        mode = self.generation_decode_mode_config_
        if mode not in valid_modes:
            raise ValueError("generation_decode_mode must be one of 'auto', 'direct', 'factorised', or 'joint'")
        if mode == "joint":
            if not hasattr(self, "joint_decoder_") or not self.joint_decoder_.can_predict_joint():
                raise ValueError("generation_decode_mode='joint' requires a fitted neural joint row decoder")
            return "joint"
        has_stochastic = self._has_stochastic_decoders()
        if mode == "factorised" and not has_stochastic:
            raise ValueError(
                "generation_decode_mode='factorised' requires a decoder that supports conditional sampling."
            )
        if mode == "auto":
            return "factorised" if has_stochastic else "direct"
        return mode

    def _task_for(self, column: str):
        if column in self.regression_columns_:
            return "regression"
        if column in self.classification_columns_:
            return "classification"
        raise KeyError(column)

    def _selected_columns(self, columns, only_missing: bool, X: pd.DataFrame):
        if columns is None:
            cols = [c for c in self.model_columns_ if (not only_missing or X[c].isna().any())]
        else:
            cols = list(columns)
        unknown = set(cols) - set(self.model_columns_)
        if unknown:
            raise ValueError(f"Unknown modelled columns: {sorted(unknown)}")
        return cols

    def _fit_neighbours(self, H):
        n = len(H)
        k = min(max(2, self.policy_.n_neighbors + 1), n)
        model = NearestNeighbors(n_neighbors=k)
        model.fit(H)
        return model

    def _condition_mask(self, condition):
        if condition is None:
            return np.ones(len(self.train_X_), dtype=bool)
        if not isinstance(condition, dict):
            raise ValueError("condition must be a mapping of column names to required values")
        mask = np.ones(len(self.train_X_), dtype=bool)
        for column, value in condition.items():
            if column not in self.train_X_.columns:
                raise ValueError(f"Unknown condition column: {column!r}")
            mask &= self.train_X_[column].astype(object).eq(value).to_numpy()
        return mask

    def _choose_neighbour(self, pos: int, rng, condition_mask=None):
        distances, indices = self.neighbour_index_.kneighbors(self.train_embeddings_[[pos]], return_distance=True)
        candidates = [int(i) for i in indices[0] if int(i) != pos]
        if condition_mask is not None:
            conditioned = [c for c in candidates if condition_mask[c]]
            candidates = conditioned or candidates
        if self.policy_.neighbour_mode == "mutual":
            mutual = []
            for c in candidates:
                _d, rev = self.neighbour_index_.kneighbors(self.train_embeddings_[[c]], return_distance=True)
                if pos in [int(i) for i in rev[0]]:
                    mutual.append(c)
            candidates = mutual or candidates
        if not candidates:
            return pos
        return int(rng.choice(candidates))

    def _plot_original_matrix(self, X):
        cols = [c for c in self.model_columns_ if c in X.columns]
        prep = ContextPreprocessor(
            numeric_columns=[c for c in cols if c in self.regression_columns_],
            categorical_columns=[c for c in cols if c in self.classification_columns_],
        )
        return prep.fit(X[cols]).transform(X[cols]).toarray()

    def _classical_mds(self, X, center=None, random_state=None):
        X = np.asarray(X, dtype=float)
        if len(X) == 0:
            return np.empty((0, 2))
        c = self._resolve_center(X, center, random_state)
        Xc = X - c
        D2 = pairwise_distances(Xc, metric="euclidean", squared=True)
        n = D2.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        vals, vecs = np.linalg.eigh(B)
        order = np.argsort(vals)[::-1][:2]
        vals = np.maximum(vals[order], 0)
        coords = vecs[:, order] * np.sqrt(vals)
        if coords.shape[1] < 2:
            coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
        return coords

    def _resolve_center(self, X, center, random_state):
        if center is None:
            return np.nanmean(X, axis=0)
        if isinstance(center, str) and center == "random":
            rng = np.random.default_rng(self.random_state if random_state is None else random_state)
            return X[int(rng.integers(0, len(X)))]
        if isinstance(center, (int, np.integer)):
            return X[int(center)]
        arr = np.asarray(center, dtype=float)
        if arr.shape[0] != X.shape[1]:
            warnings.warn("Explicit center dimensionality does not match; falling back to mean", RuntimeWarning)
            return np.nanmean(X, axis=0)
        return arr

    def _plot_colors(self, X, color_by):
        if color_by is None:
            return None, False
        if color_by not in X.columns:
            raise ValueError(f"color_by column {color_by!r} is not in X")
        series = X[color_by]
        return series, pd.api.types.is_numeric_dtype(series)

    def _as_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def _to_2d(self, X, width: int | None = None):
        arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if width is not None:
            if arr.shape[1] < width:
                arr = np.pad(arr, ((0, 0), (0, width - arr.shape[1])))
            elif arr.shape[1] > width:
                arr = arr[:, :width]
        return arr

    def _residual(self, observed, prediction, task):
        if pd.isna(observed):
            return np.nan
        if task == "regression":
            return float(observed) - float(prediction)
        return np.nan

    def _discrepancy(self, observed, prediction, task):
        if pd.isna(observed):
            return np.nan
        if task == "regression":
            return abs(float(observed) - float(prediction))
        return float(observed != prediction)

    def _decoder_name(self):
        decoder = getattr(self, "decoder_", self.decoder)
        if decoder is None:
            return "MixedFeatureDecoder.random_forest"
        return decoder.__class__.__name__

    def _verbose_init(self):
        print("MIMIC init hyperparameters:")
        preview = self._verbose_configuration_preview()
        for key, value in preview.items():
            self._verbose_print_item(key, value)

    def _verbose_fit_configuration(self):
        if not self.verbose:
            return
        print("MIMIC resolved fit configuration:")
        print(f"  mode_: {self.mode_}")
        print(f"  level_: {self.level_}")
        print(f"  capacity_: {self.capacity_}")
        print(f"  bootstrap_: {self.bootstrap_}")
        print(f"  n_bootstrap_: {self.n_bootstrap_}")
        print(f"  generation_decode_mode_config_: {self.generation_decode_mode_config_}")
        self._verbose_print_item("encoder_", self.encoder_)
        self._verbose_print_item("decoder_", self.decoder_)
        self._verbose_print_item("policy_config_", self.policy_config_)

    def _verbose_fit_summary(self):
        if not self.verbose:
            return
        print("MIMIC fitted data sizes:")
        print(f"  input_rows: {len(self.train_X_)}")
        print(f"  input_columns: {len(self.columns_)}")
        print(f"  ignored_columns: {len(self.ignore_columns_)}")
        print(f"  regression_columns: {len(self.regression_columns_)}")
        print(f"  classification_columns: {len(self.classification_columns_)}")
        print(f"  model_columns: {len(self.model_columns_)}")
        print(f"  encoded_input_dim: {self.global_preprocessor_.output_dim_}")
        print(f"  train_embeddings_shape: {self.train_embeddings_.shape}")
        print(f"  resolved_generation_decode_mode: {self.generation_decode_mode_}")
        for column, sl in self.embedding_slices_.items():
            print(f"  embedding[{column}]: rows={self.train_embeddings_.shape[0]}, dim={sl.stop - sl.start}")

    @staticmethod
    def _verbose_value(value):
        if value is None or isinstance(value, (str, int, float, bool, tuple, list)):
            return value
        return value.__class__.__name__

    def _verbose_configuration_preview(self):
        params = dict(self.get_params(deep=False))
        try:
            mode_name = self._resolve_mode_name()
            capacity = self._validate_capacity(self.capacity)
            preset_params = self._capacity_parameters(capacity)
            _, _, preset_mode = self._preset_components(mode_name, preset_params)
            params["mode"] = mode_name
            params["capacity"] = capacity
            if self.generation_decode_mode == "auto" and mode_name is not None and self.decoder is None:
                params["generation_decode_mode"] = preset_mode
            else:
                params["generation_decode_mode"] = self.generation_decode_mode
        except ValueError:
            return params
        return params

    def _preset_components(self, mode_name, preset_params):
        if mode_name == "identity":
            return IdentityEncoder(), IdentityDecoder(), "direct"
        if mode_name == "direct":
            return self._capacity_resnet_encoder(preset_params), self._capacity_neural_decoder(preset_params), "direct"
        if mode_name == "factorised":
            return self._capacity_resnet_encoder(preset_params), self._capacity_neural_decoder(preset_params), "factorised"
        if mode_name == "joint":
            return self._capacity_resnet_encoder(preset_params), self._capacity_neural_decoder(preset_params), "joint"
        return (
            RandomForestPathEncoder(n_estimators=50, n_jobs=self.n_jobs),
            MixedFeatureDecoder.random_forest(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            "auto",
        )

    def _default_generation_policy(self):
        return GenerationPolicy(
            method="displacement",
            neighbour_mode="mutual",
            n_neighbors=5,
            lambda_range=(0.25, 0.75),
        )

    @classmethod
    def _verbose_print_item(cls, key, value, indent: int = 2):
        prefix = " " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            for child_key, child_value in value.items():
                cls._verbose_print_item(child_key, child_value, indent=indent + 2)
        elif cls._is_verbose_config_object(value):
            print(f"{prefix}{key}:")
            print(f"{prefix}  class: {value.__class__.__name__}")
            print(f"{prefix}  parameters:")
            for child_key, child_value in cls._verbose_object_parameters(value).items():
                cls._verbose_print_item(child_key, child_value, indent=indent + 4)
        elif isinstance(value, list):
            print(f"{prefix}{key}:")
            if not value:
                print(f"{prefix}  []")
            for item in value:
                print(f"{prefix}  - {cls._verbose_value(item)}")
        else:
            print(f"{prefix}{key}: {cls._verbose_value(value)}")

    @staticmethod
    def _is_verbose_config_object(value):
        return value is not None and (hasattr(value, "get_params") or is_dataclass(value))

    @staticmethod
    def _verbose_object_parameters(value):
        if hasattr(value, "get_params"):
            return dict(value.get_params(deep=False))
        if is_dataclass(value):
            return {field.name: getattr(value, field.name) for field in fields(value)}
        return {}


def mimic_data(
    df: pd.DataFrame,
    n_samples: int | None = None,
    *,
    mode="factorised",
    capacity: float = 0.25,
    save_model=None,
    load_model=None,
    refit: bool = False,
    privacy_filter=None,
    **mimic_kwargs,
) -> pd.DataFrame:
    """Fit a default MIMIC model and return synthetic rows for ``df``.

    This is the smallest public interface for one-shot generation. By default
    it returns the same number of rows as the input dataframe. Additional
    keyword arguments are passed to ``MIMIC`` for callers who need to override
    defaults such as ``random_state`` or ``columns``. Pass ``save_model`` to
    persist a freshly fitted model, or ``load_model`` to sample from a fitted
    local artifact without retraining. Use ``refit=True`` to fit a fresh model
    even when ``load_model`` is supplied.
    """

    if load_model is not None and not refit:
        model = MIMIC.load(load_model)
    else:
        model = MIMIC(mode=mode, capacity=capacity, **mimic_kwargs)
        model.fit(df)
        if save_model is not None:
            model.save(save_model)
    return model.sample(len(df) if n_samples is None else n_samples, privacy_filter=privacy_filter)


sample = mimic_data
sample_dataframe = mimic_data
