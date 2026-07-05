"""Main MIMIC estimator."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import json
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .decoders import IdentityDecoder, MixedFeatureDecoder, NeuralConditionalSampler
from .encoders import IdentityEncoder, RandomForestPathEncoder, ResNetEncoder
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
class BootstrapMember:
    preprocessor: ContextPreprocessor
    encoder: object
    decoder: MixedFeatureDecoder
    embedding_dim: int


@dataclass
class FeatureModule:
    target_column: str
    task: str
    context_columns: list[str]
    members: list[BootstrapMember]
    observed_mask: pd.Series
    label_encoder: LabelEncoder | None = None
    target_scaler: StandardScaler | None = None
    bias: float | None = None
    noise: float | None = None
    classes_: np.ndarray | None = None


class MIMIC(BaseEstimator, TransformerMixin):
    """Modular feature-wise estimator for imputation, confidence, and generation."""

    def __init__(
        self,
        columns=None,
        encoder=None,
        decoder=None,
        policy=None,
        generation_decode_mode: str = "auto",
        mode="joint",
        level=None,
        capacity: float = 0.5,
        n_bootstrap: int | None = None,
        random_state: int | None = None,
        n_jobs: int | None = None,
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
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = self._as_dataframe(X).copy()
        self._validate_schema(X)
        self._resolve_mode_configuration()
        rng = np.random.default_rng(self.random_state)
        self.train_X_ = X.copy()
        self.train_index_ = X.index.copy()
        self.input_dtypes_ = X.dtypes.to_dict()
        self.feature_modules_ = {}

        for column in self.model_columns_:
            task = self._task_for(column)
            observed_mask = X[column].notna()
            observed_indices = np.flatnonzero(observed_mask.to_numpy())
            if len(observed_indices) < 2:
                raise ValueError(f"Column {column!r} has too few observed rows")
            if task == "classification" and X.loc[observed_mask, column].nunique() < 2:
                raise ValueError(f"Classification column {column!r} needs at least two observed classes")

            context_columns = self._context_columns_for(column)
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
            for b in range(self.n_bootstrap_):
                sample_pos = rng.choice(np.arange(len(observed_indices)), size=len(observed_indices), replace=True)
                sample_rows = observed_indices[sample_pos]
                sampled_observed_positions = sample_pos
                sampled_y = y_full[sampled_observed_positions]

                X_context_sample = X.iloc[sample_rows][context_columns]
                preprocessor = self._new_context_preprocessor(context_columns)
                Xp = preprocessor.fit(X_context_sample).transform(X_context_sample)

                encoder = self._new_encoder(task, b)
                encoder.fit(Xp, sampled_y)
                H = self._to_2d(encoder.transform(Xp))

                decoder = self._new_decoder()
                decoder.fit_target(column, task, H, sampled_y)
                members.append(
                    BootstrapMember(
                        preprocessor=preprocessor,
                        encoder=encoder,
                        decoder=decoder,
                        embedding_dim=H.shape[1],
                    )
                )

                oob_source = np.setdiff1d(np.arange(len(observed_indices)), np.unique(sampled_observed_positions))
                if len(oob_source):
                    oob_rows = observed_indices[oob_source]
                    Xp_oob = preprocessor.transform(X.iloc[oob_rows][context_columns])
                    H_oob = self._to_2d(encoder.transform(Xp_oob), width=H.shape[1])
                    pred = decoder.predict_target(column, H_oob)
                    if task == "regression":
                        pred_raw = self._inverse_regression_target(pred.astype(float), target_scaler)
                        true_raw = self._inverse_regression_target(y_full[oob_source].astype(float), target_scaler)
                        err = pred_raw - true_raw
                        oob_errors.extend(err.tolist())

            bias = float(np.mean(oob_errors)) if oob_errors else 0.0
            noise = float(np.var(oob_errors, ddof=1)) if len(oob_errors) > 1 else 0.0
            self.feature_modules_[column] = FeatureModule(
                target_column=column,
                task=task,
                context_columns=context_columns,
                members=members,
                observed_mask=observed_mask,
                label_encoder=label_encoder,
                target_scaler=target_scaler,
                bias=bias,
                noise=noise,
                classes_=classes,
            )

        self.train_embeddings_ = self.transform(X)
        self.embedding_slices_ = self._embedding_slices_
        self._fit_conditional_samplers()
        if self.generation_decode_mode_config_ == "joint":
            self._fit_joint_decoder()
        self.generation_decode_mode_ = self._resolve_generation_decode_mode()
        self.policy_ = self._new_policy()
        self.policy_.validate()
        self.neighbour_index_ = self._fit_neighbours(self.train_embeddings_)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_modules_")
        X = self._as_dataframe(X)
        parts = []
        self._embedding_slices_ = {}
        start = 0
        for column, module in self.feature_modules_.items():
            embeddings = []
            max_width = max(member.embedding_dim for member in module.members)
            for member in module.members:
                Xp = member.preprocessor.transform(X[module.context_columns])
                emb = self._to_2d(member.encoder.transform(Xp), width=max_width)
                embeddings.append(emb)
            block = np.mean(np.stack(embeddings, axis=0), axis=0)
            parts.append(block)
            self._embedding_slices_[column] = slice(start, start + block.shape[1])
            start += block.shape[1]
        return np.hstack(parts) if parts else np.empty((len(X), 0))

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
                rows.append(base)
        return pd.DataFrame(rows)

    def sample(self, n_samples: int, condition=None, return_trace: bool = False):
        check_is_fitted(self, "train_embeddings_")
        rng = np.random.default_rng(self.random_state)
        H = np.asarray(self.train_embeddings_)
        condition_mask = self._condition_mask(condition)
        anchor_candidates = np.flatnonzero(condition_mask)
        if len(anchor_candidates) == 0:
            raise ValueError("No training rows satisfy the requested generation condition")
        synth_embeddings = []
        traces = []
        for sample_index in range(n_samples):
            anchor_pos = int(rng.choice(anchor_candidates))
            neigh_pos = self._choose_neighbour(anchor_pos, rng, condition_mask=condition_mask)
            lam = float(rng.uniform(*self.policy_.lambda_range))
            if self.policy_.method == "smote":
                h_new = (1.0 - lam) * H[anchor_pos] + lam * H[neigh_pos]
                trace = {
                    "trace_type": "embedding",
                    "sample_index": sample_index,
                    "method": "smote",
                    "anchor_index": self.train_index_[anchor_pos],
                    "neighbour_index": self.train_index_[neigh_pos],
                    "lambda": lam,
                    "neighbour_mode": self.policy_.neighbour_mode,
                    "condition": json.dumps(condition, sort_keys=True) if condition is not None else None,
                    "decoder": self._decoder_name(),
                    "generation_decode_mode": self.generation_decode_mode,
                    "resolved_generation_decode_mode": self.generation_decode_mode_,
                    "random_state": self.random_state,
                }
            else:
                from_pos = neigh_pos
                to_pos = self._choose_neighbour(from_pos, rng, condition_mask=condition_mask)
                h_new = H[anchor_pos] + lam * (H[to_pos] - H[from_pos])
                trace = {
                    "trace_type": "embedding",
                    "sample_index": sample_index,
                    "method": "displacement",
                    "anchor_index": self.train_index_[anchor_pos],
                    "displacement_from_index": self.train_index_[from_pos],
                    "displacement_to_index": self.train_index_[to_pos],
                    "lambda": lam,
                    "neighbour_mode": self.policy_.neighbour_mode,
                    "restriction": "basic",
                    "condition": json.dumps(condition, sort_keys=True) if condition is not None else None,
                    "decoder": self._decoder_name(),
                    "generation_decode_mode": self.generation_decode_mode,
                    "resolved_generation_decode_mode": self.generation_decode_mode_,
                    "random_state": self.random_state,
                }
            synth_embeddings.append(h_new)
            traces.append(trace)

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
            X_new, cell_traces = self._gibbs_sample_decoded(
                H_new,
                X_new,
                condition=condition,
                rng=rng,
                n_sweeps=3,
                return_trace=return_trace,
            )
            traces.extend(cell_traces)
        X_new = self._restore_sample_schema(X_new)
        if return_trace:
            return X_new, pd.DataFrame(traces)
        return X_new

    def plot(self, X=None, color_by=None, center=None, random_state=None, ax=None):
        check_is_fitted(self, "feature_modules_")
        import matplotlib.pyplot as plt

        X = self.train_X_ if X is None else self._as_dataframe(X)
        original = self._plot_original_matrix(X)
        embedding = self.transform(X)
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

    def _predict_column(self, X: pd.DataFrame, column: str):
        module = self.feature_modules_[column]
        if module.task == "regression":
            preds = []
            max_width = max(member.embedding_dim for member in module.members)
            for member in module.members:
                H = self._member_embedding(member, module, X, max_width=member.embedding_dim)
                pred_scaled = member.decoder.predict_target(column, H).astype(float)
                preds.append(self._inverse_regression_target(pred_scaled, module.target_scaler))
            arr = np.vstack(preds)
            mean = arr.mean(axis=0)
            variance = arr.var(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
            details = {
                "variance": variance,
                "uncertainty": variance + (module.noise or 0.0),
            }
            return mean, details

        probas = []
        votes = []
        for member in module.members:
            H = self._member_embedding(member, module, X, max_width=member.embedding_dim)
            p = self._aligned_predict_proba(member, module, column, H)
            probas.append(p)
            votes.append(np.argmax(p, axis=1))
        p_arr = np.stack(probas, axis=0)
        p_mean = p_arr.mean(axis=0)
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

    def _member_embedding(self, member: BootstrapMember, module: FeatureModule, X: pd.DataFrame, max_width: int | None = None):
        Xp = member.preprocessor.transform(X[module.context_columns])
        return self._to_2d(member.encoder.transform(Xp), width=max_width)

    def _decode_embeddings(self, H):
        data = {}
        for column, module in self.feature_modules_.items():
            sl = self.embedding_slices_[column]
            block = H[:, sl]
            if module.task == "regression":
                preds = []
                for member in module.members:
                    b = self._to_2d(block, width=member.embedding_dim)[:, : member.embedding_dim]
                    pred_scaled = member.decoder.predict_target(column, b).astype(float)
                    preds.append(self._inverse_regression_target(pred_scaled, module.target_scaler))
                data[column] = np.vstack(preds).mean(axis=0)
            else:
                probas = []
                for member in module.members:
                    b = self._to_2d(block, width=member.embedding_dim)[:, : member.embedding_dim]
                    probas.append(self._aligned_predict_proba(member, module, column, b))
                pred_codes = np.argmax(np.stack(probas, axis=0).mean(axis=0), axis=1)
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
            for member in module.members:
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
            for member in module.members:
                decoder = member.decoder
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
            member_evidence = []
            for member in module.members:
                if not hasattr(member.decoder, "conditional_evidence_target"):
                    raise ValueError(
                        "generation_decode_mode='joint' requires NeuralConditionalSampler-style conditional evidence"
                    )
                member_evidence.append(member.decoder.conditional_evidence_target(column, module.task, H_context))
            blocks.append(np.mean(np.stack(member_evidence, axis=0), axis=0))
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
            for member in module.members:
                if hasattr(member.decoder, "can_sample_target") and member.decoder.can_sample_target(column):
                    return True
        return False

    def _gibbs_sample_decoded(self, H, X, condition, rng, n_sweeps: int, return_trace: bool):
        X = X.copy()
        condition = condition or {}
        traces = []
        for sweep in range(n_sweeps):
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
                            "method": "gibbs_forest",
                            "sweep": sweep,
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
        candidates = [
            (i, member)
            for i, member in enumerate(module.members)
            if hasattr(member.decoder, "can_sample_target") and member.decoder.can_sample_target(column)
        ]
        if not candidates:
            return None, None
        return candidates[int(rng.integers(0, len(candidates)))]

    @staticmethod
    def _without_slice(H, sl):
        return np.hstack([H[:, : sl.start], H[:, sl.stop :]])

    def _aligned_predict_proba(self, member: BootstrapMember, module: FeatureModule, column: str, H):
        proba = member.decoder.predict_proba_target(column, H)
        model = member.decoder.models_[column]
        model_classes = getattr(model, "classes_", np.arange(proba.shape[1]))
        n_classes = len(module.classes_)
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
        if self.columns is None:
            self.ignore_columns_ = []
            model_cols = [c for c in X.columns if c not in self.ignore_columns_]
            self.regression_columns_ = [c for c in model_cols if pd.api.types.is_numeric_dtype(X[c])]
            self.classification_columns_ = [c for c in model_cols if c not in self.regression_columns_]
        else:
            if not isinstance(self.columns, dict):
                raise ValueError("columns must be a mapping with 'ignore', 'regression', and 'classification' lists")
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
        mode_name = self._resolve_mode_name()
        capacity = self._validate_capacity(self.capacity)
        preset_params = self._capacity_parameters(capacity)
        self.mode_ = mode_name
        self.level_ = self._mode_to_level(mode_name)
        self.capacity_ = capacity
        self.capacity_parameters_ = preset_params

        self.n_bootstrap_ = preset_params["n_bootstrap"] if self.n_bootstrap is None else self.n_bootstrap
        self.policy_config_ = self.policy if self.policy is not None else GenerationPolicy(
            method="displacement",
            neighbour_mode="mutual",
            n_neighbors=5,
            lambda_range=(0.25, 0.75),
        )

        if mode_name == "identity":
            preset_encoder = IdentityEncoder()
            preset_decoder = IdentityDecoder()
            preset_mode = "direct"
        elif mode_name == "direct":
            preset_encoder = self._capacity_resnet_encoder(preset_params)
            preset_decoder = self._capacity_neural_decoder(preset_params)
            preset_mode = "direct"
        elif mode_name == "factorised":
            preset_encoder = self._capacity_resnet_encoder(preset_params)
            preset_decoder = self._capacity_neural_decoder(preset_params)
            preset_mode = "factorised"
        elif mode_name == "joint":
            preset_encoder = self._capacity_resnet_encoder(preset_params)
            preset_decoder = self._capacity_neural_decoder(preset_params)
            preset_mode = "joint"
        else:
            preset_encoder = RandomForestPathEncoder(n_estimators=50, n_jobs=self.n_jobs)
            preset_decoder = MixedFeatureDecoder.random_forest(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            preset_mode = "auto"

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
