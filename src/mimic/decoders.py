"""Decoder implementations for MIMIC."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.validation import check_is_fitted

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - import availability is environment dependent
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None


class MixedFeatureDecoder(BaseEstimator):
    """Combined decoder that fits one estimator per target feature."""

    def __init__(self, regression_estimator=None, classification_estimator=None):
        self.regression_estimator = regression_estimator
        self.classification_estimator = classification_estimator

    @classmethod
    def linear(cls):
        return LinearMixedFeatureDecoder()

    @classmethod
    def random_forest(cls, n_estimators: int = 100, random_state: int | None = None, n_jobs: int | None = None):
        return cls(
            regression_estimator=RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=n_jobs,
            ),
            classification_estimator=RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        )

    def fit(self, X, y=None):
        self.models_ = {}
        self.tasks_ = {}
        return self

    def fit_target(self, column: str, task: str, H, y):
        if not hasattr(self, "models_"):
            self.models_ = {}
            self.tasks_ = {}
        estimator = self._estimator_for_task(task)
        model = clone(estimator)
        model.fit(H, y)
        if task == "classification" and not hasattr(model, "predict_proba"):
            raise ValueError(f"Classification decoder for {column!r} must implement predict_proba")
        self.models_[column] = model
        self.tasks_[column] = task
        return self

    def predict_target(self, column: str, H):
        check_is_fitted(self, "models_")
        return self.models_[column].predict(H)

    def predict_proba_target(self, column: str, H):
        check_is_fitted(self, "models_")
        model = self.models_[column]
        if not hasattr(model, "predict_proba"):
            raise ValueError(f"Decoder for {column!r} does not implement predict_proba")
        return model.predict_proba(H)

    def decode(self, H, columns=None):
        check_is_fitted(self, "models_")
        columns = list(self.models_) if columns is None else list(columns)
        data = {column: self.predict_target(column, H) for column in columns}
        return pd.DataFrame(data)

    def can_sample_target(self, column: str) -> bool:
        return False

    def _estimator_for_task(self, task: str):
        if task == "regression":
            if self.regression_estimator is None:
                return RandomForestRegressor(n_estimators=100)
            return self.regression_estimator
        if task == "classification":
            if self.classification_estimator is None:
                return RandomForestClassifier(n_estimators=100)
            return self.classification_estimator
        raise ValueError("task must be 'regression' or 'classification'")


class IdentityDecoder(MixedFeatureDecoder):
    """Decoder for identity/preprocessed-space generation baselines.

    For regression targets, the decoder selects the embedding coordinate most
    correlated with the scaled target. For classification targets, it selects
    one coordinate per class, which corresponds to the target one-hot columns
    when used with ``IdentityEncoder``.
    """

    def __init__(self):
        super().__init__(regression_estimator=None, classification_estimator=None)

    def fit(self, X, y=None):
        self.models_ = {}
        self.tasks_ = {}
        return self

    def fit_target(self, column: str, task: str, H, y):
        if not hasattr(self, "models_"):
            self.models_ = {}
            self.tasks_ = {}
        H = self._as_dense(H)
        y_arr = np.asarray(y)
        if task == "regression":
            model = self._fit_regression_identity(H, y_arr)
        elif task == "classification":
            model = self._fit_classification_identity(H, y_arr)
        else:
            raise ValueError("task must be 'regression' or 'classification'")
        self.models_[column] = model
        self.tasks_[column] = task
        return self

    def predict_target(self, column: str, H):
        check_is_fitted(self, "models_")
        H = self._as_dense(H)
        model = self.models_[column]
        if self.tasks_[column] == "regression":
            return H[:, model["feature_index"]]
        proba = self._classification_scores(model, H)
        return model["classes"][np.argmax(proba, axis=1)]

    def predict_proba_target(self, column: str, H):
        check_is_fitted(self, "models_")
        if self.tasks_[column] != "classification":
            raise ValueError(f"Decoder for {column!r} is not a classification decoder")
        return self._classification_scores(self.models_[column], self._as_dense(H))

    def decode(self, H, columns=None):
        check_is_fitted(self, "models_")
        columns = list(self.models_) if columns is None else list(columns)
        data = {column: self.predict_target(column, H) for column in columns}
        return pd.DataFrame(data)

    @staticmethod
    def _fit_regression_identity(H, y):
        y = np.asarray(y, dtype=float)
        y_centered = y - np.nanmean(y)
        scores = []
        for j in range(H.shape[1]):
            x = np.asarray(H[:, j], dtype=float)
            x_centered = x - np.nanmean(x)
            denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
            scores.append(0.0 if denom == 0 else abs(float(np.dot(x_centered, y_centered) / denom)))
        return {"feature_index": int(np.argmax(scores))}

    @staticmethod
    def _fit_classification_identity(H, y):
        classes = np.unique(y)
        feature_indices = []
        for cls in classes:
            y_bin = (y == cls).astype(float)
            if y_bin.max() == y_bin.min():
                feature_indices.append(0)
                continue
            scores = []
            for j in range(H.shape[1]):
                x = np.asarray(H[:, j], dtype=float)
                x_centered = x - np.nanmean(x)
                y_centered = y_bin - np.nanmean(y_bin)
                denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
                scores.append(0.0 if denom == 0 else float(np.dot(x_centered, y_centered) / denom))
            feature_indices.append(int(np.argmax(scores)))
        return {"classes": classes, "feature_indices": np.asarray(feature_indices, dtype=int)}

    @staticmethod
    def _classification_scores(model, H):
        scores = np.asarray(H[:, model["feature_indices"]], dtype=float)
        scores = scores - np.nanmax(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        row_sums = exp_scores.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return exp_scores / row_sums

    @staticmethod
    def _as_dense(H):
        if hasattr(H, "toarray"):
            return H.toarray()
        arr = np.asarray(H)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr


def mean_regression_prediction(predictions):
    arr = np.asarray(predictions, dtype=float)
    return np.nanmean(arr, axis=0)


class LinearMixedFeatureDecoder(MixedFeatureDecoder):
    """Mixed decoder backed by scikit-learn linear models.

    Regression targets use ``LinearRegression``. Classification targets use
    ``LogisticRegression`` so that ``predict_proba`` is available for MIMIC's
    confidence diagnostics.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        positive: bool = False,
        logistic_C: float = 1.0,
        logistic_penalty: str = "l2",
        logistic_solver: str = "lbfgs",
        logistic_max_iter: int = 1000,
        logistic_class_weight=None,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ):
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.logistic_C = logistic_C
        self.logistic_penalty = logistic_penalty
        self.logistic_solver = logistic_solver
        self.logistic_max_iter = logistic_max_iter
        self.logistic_class_weight = logistic_class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__(regression_estimator=None, classification_estimator=None)

    def _estimator_for_task(self, task: str):
        if task == "regression":
            return LinearRegression(
                fit_intercept=self.fit_intercept,
                positive=self.positive,
                n_jobs=self.n_jobs,
            )
        if task == "classification":
            return LogisticRegression(
                C=self.logistic_C,
                penalty=self.logistic_penalty,
                solver=self.logistic_solver,
                max_iter=self.logistic_max_iter,
                class_weight=self.logistic_class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        raise ValueError("task must be 'regression' or 'classification'")


class ForestConditionalSampler(MixedFeatureDecoder):
    """Random-forest decoder with stochastic conditional sampling support.

    Deterministic predictions use ordinary random-forest regressors/classifiers.
    Stochastic sampling uses separate target models fit on ``z_-j`` contexts:
    classifiers sample from ``predict_proba``; regressors sample exact observed
    target values using forest leaf weights.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int | None = None,
        n_jobs: int | None = None,
        min_samples_leaf: int = 1,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.min_samples_leaf = min_samples_leaf
        super().__init__(regression_estimator=None, classification_estimator=None)

    def _estimator_for_task(self, task: str):
        if task == "regression":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                min_samples_leaf=self.min_samples_leaf,
            )
        if task == "classification":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                min_samples_leaf=self.min_samples_leaf,
            )
        raise ValueError("task must be 'regression' or 'classification'")

    def fit(self, X, y=None):
        super().fit(X, y)
        self.sampler_models_ = {}
        self.sampler_tasks_ = {}
        self.sampler_targets_ = {}
        self.sampler_train_indices_ = {}
        self.sampler_train_leaves_ = {}
        return self

    def fit_target(self, column: str, task: str, H, y):
        super().fit_target(column, task, H, y)
        if not hasattr(self, "sampler_models_"):
            self.sampler_models_ = {}
            self.sampler_tasks_ = {}
            self.sampler_targets_ = {}
            self.sampler_train_indices_ = {}
            self.sampler_train_leaves_ = {}
        return self

    def fit_sampler_target(self, column: str, task: str, H_context, y, train_indices=None):
        if not hasattr(self, "sampler_models_"):
            self.sampler_models_ = {}
            self.sampler_tasks_ = {}
            self.sampler_targets_ = {}
            self.sampler_train_indices_ = {}
            self.sampler_train_leaves_ = {}
        model = self._estimator_for_task(task)
        model.fit(H_context, y)
        self.sampler_models_[column] = model
        self.sampler_tasks_[column] = task
        self.sampler_targets_[column] = np.asarray(y)
        self.sampler_train_indices_[column] = None if train_indices is None else np.asarray(train_indices)
        if task == "regression":
            self.sampler_train_leaves_[column] = model.apply(H_context)
        return self

    def can_sample_target(self, column: str) -> bool:
        return hasattr(self, "sampler_models_") and column in self.sampler_models_

    def sample_target(self, column: str, task: str, H_context, rng, return_trace: bool = False):
        check_is_fitted(self, "sampler_models_")
        if column not in self.sampler_models_:
            raise ValueError(f"No sampler has been fit for target {column!r}")
        if task == "classification":
            return self._sample_classification(column, H_context, rng, return_trace=return_trace)
        if task == "regression":
            return self._sample_regression(column, H_context, rng, return_trace=return_trace)
        raise ValueError("task must be 'regression' or 'classification'")

    def _sample_classification(self, column: str, H_context, rng, return_trace: bool):
        model = self.sampler_models_[column]
        proba = model.predict_proba(H_context)
        classes = np.asarray(model.classes_)
        values = []
        traces = []
        for row_proba in proba:
            p = np.asarray(row_proba, dtype=float)
            p = p / p.sum() if p.sum() else np.ones_like(p) / len(p)
            chosen_pos = int(rng.choice(np.arange(len(classes)), p=p))
            values.append(classes[chosen_pos])
            if return_trace:
                traces.append(
                    {
                        "sampled_class_code": int(classes[chosen_pos]) if np.issubdtype(classes.dtype, np.integer) else classes[chosen_pos],
                        "class_probabilities": {str(classes[i]): float(p[i]) for i in range(len(classes))},
                    }
                )
        values = np.asarray(values)
        return (values, traces) if return_trace else values

    def _sample_regression(self, column: str, H_context, rng, return_trace: bool):
        model = self.sampler_models_[column]
        train_leaves = self.sampler_train_leaves_[column]
        targets = self.sampler_targets_[column]
        train_indices = self.sampler_train_indices_[column]
        query_leaves = model.apply(H_context)
        values = []
        traces = []
        for leaves in query_leaves:
            weights = self._leaf_weights(leaves, train_leaves)
            source_pos = int(rng.choice(np.arange(len(targets)), p=weights))
            values.append(targets[source_pos])
            if return_trace:
                support = int(np.count_nonzero(weights))
                traces.append(
                    {
                        "source_position": source_pos,
                        "source_index": self._python_scalar(None if train_indices is None else train_indices[source_pos]),
                        "source_weight": float(weights[source_pos]),
                        "leaf_support_size": support,
                    }
                )
        values = np.asarray(values, dtype=float)
        return (values, traces) if return_trace else values

    @staticmethod
    def _leaf_weights(query_leaves, train_leaves):
        weights = np.zeros(train_leaves.shape[0], dtype=float)
        for tree_idx, leaf in enumerate(query_leaves):
            matches = train_leaves[:, tree_idx] == leaf
            count = int(matches.sum())
            if count:
                weights[matches] += 1.0 / (train_leaves.shape[1] * count)
        total = weights.sum()
        if total <= 0:
            weights[:] = 1.0 / len(weights)
        else:
            weights /= total
        return weights

    @staticmethod
    def _python_scalar(value):
        if value is None:
            return None
        return value.item() if hasattr(value, "item") else value


if nn is not None:

    class _NeuralSamplerNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float, out_dim: int):
            super().__init__()
            layers = []
            width = input_dim
            for _ in range(max(1, n_layers)):
                layers.extend(
                    [
                        nn.Linear(width, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                width = hidden_dim
            layers.append(nn.Linear(width, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


def mdn_negative_log_likelihood(logits, mu, raw_sigma, y, min_sigma: float = 1e-3, max_sigma: float = 5.0):
    """Return mean negative log likelihood under a Gaussian mixture density head."""

    if torch is None:
        raise ImportError("mdn_negative_log_likelihood requires torch")
    y = y.reshape(-1, 1)
    sigma = (F.softplus(raw_sigma) + min_sigma).clamp(max=max_sigma)
    log_pi = torch.log_softmax(logits, dim=-1)
    log_prob = torch.distributions.Normal(mu, sigma).log_prob(y)
    return -torch.logsumexp(log_pi + log_prob, dim=-1).mean()


class _NeuralTargetModel(dict):
    @property
    def classes_(self):
        return self.get("classes_")


class NeuralConditionalSampler(MixedFeatureDecoder):
    """Neural decoder with MDN regression sampling and softmax classification sampling."""

    def __init__(
        self,
        n_components: int = 5,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.05,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 100,
        patience: int = 10,
        validation_fraction: float = 0.1,
        min_sigma: float = 1e-3,
        max_sigma: float = 5.0,
        component_temperature: float = 1.0,
        noise_scale: float = 1.0,
        gradient_clip: float = 5.0,
        random_state: int | None = None,
        device: str = "auto",
    ):
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.component_temperature = component_temperature
        self.noise_scale = noise_scale
        self.gradient_clip = gradient_clip
        self.random_state = random_state
        self.device = device
        super().__init__(regression_estimator=None, classification_estimator=None)

    def fit(self, X, y=None):
        self.models_ = {}
        self.tasks_ = {}
        self.sampler_models_ = {}
        self.sampler_tasks_ = {}
        return self

    def fit_target(self, column: str, task: str, H, y):
        self._require_torch()
        if not hasattr(self, "models_"):
            self.fit(None)
        model = self._fit_neural_model(task, H, y)
        self.models_[column] = model
        self.tasks_[column] = task
        return self

    def fit_sampler_target(self, column: str, task: str, H_context, y, train_indices=None):
        self._require_torch()
        if not hasattr(self, "sampler_models_"):
            self.sampler_models_ = {}
            self.sampler_tasks_ = {}
        model = self._fit_neural_model(task, H_context, y)
        self.sampler_models_[column] = model
        self.sampler_tasks_[column] = task
        return self

    def predict_target(self, column: str, H):
        check_is_fitted(self, "models_")
        model = self.models_[column]
        if self.tasks_[column] == "regression":
            logits, mu, _sigma = self._predict_mdn_params(model, H)
            return (self._softmax_np(logits) * mu).sum(axis=1)
        proba = self._predict_class_proba(model, H)
        return model["classes"][np.argmax(proba, axis=1)]

    def predict_proba_target(self, column: str, H):
        check_is_fitted(self, "models_")
        if self.tasks_[column] != "classification":
            raise ValueError(f"Decoder for {column!r} is not a classification decoder")
        return self._predict_class_proba(self.models_[column], H)

    def can_sample_target(self, column: str) -> bool:
        return hasattr(self, "sampler_models_") and column in self.sampler_models_

    def sample_target(self, column: str, task: str, H_context, rng, return_trace: bool = False):
        check_is_fitted(self, "sampler_models_")
        if column not in self.sampler_models_:
            raise ValueError(f"No sampler has been fit for target {column!r}")
        if task == "classification":
            return self._sample_classification(column, H_context, rng, return_trace=return_trace)
        if task == "regression":
            return self._sample_regression(column, H_context, rng, return_trace=return_trace)
        raise ValueError("task must be 'regression' or 'classification'")

    def _fit_neural_model(self, task: str, X, y):
        X_arr = self._as_float32_array(X)
        if task == "classification":
            classes, y_arr = np.unique(y, return_inverse=True)
            y_tensor = torch.as_tensor(y_arr, dtype=torch.long)
            out_dim = len(classes)
        elif task == "regression":
            classes = None
            y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
            y_tensor = torch.as_tensor(y_arr, dtype=torch.float32)
            out_dim = self.n_components * 3
        else:
            raise ValueError("task must be 'regression' or 'classification'")

        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        rng = np.random.default_rng(self.random_state)
        indices = np.arange(len(X_arr))
        rng.shuffle(indices)
        n_val = int(round(len(indices) * self.validation_fraction))
        if n_val > 0 and len(indices) - n_val >= 2:
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
        else:
            val_idx = np.array([], dtype=int)
            train_idx = indices

        device = self._resolve_device()
        net = _NeuralSamplerNet(
            input_dim=X_arr.shape[1],
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            out_dim=out_dim,
        ).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        train_ds = TensorDataset(torch.as_tensor(X_arr[train_idx], dtype=torch.float32), y_tensor[train_idx])
        loader = DataLoader(train_ds, batch_size=min(self.batch_size, len(train_ds)), shuffle=True)

        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        for _epoch in range(self.max_epochs):
            net.train()
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                loss = self._training_loss(net(xb), yb, task)
                loss.backward()
                if self.gradient_clip is not None and self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), self.gradient_clip)
                optimizer.step()

            eval_idx = val_idx if len(val_idx) else train_idx
            loss_value = self._eval_loss(net, X_arr[eval_idx], y_tensor[eval_idx], task, device)
            if loss_value < best_loss - 1e-8:
                best_loss = loss_value
                best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        model = _NeuralTargetModel({
            "task": task,
            "network": net,
            "device": device,
            "classes": classes,
            "classes_": classes,
        })
        return model

    def _training_loss(self, raw, y, task: str):
        if task == "classification":
            return F.cross_entropy(raw, y)
        logits, mu, raw_sigma = self._split_mdn_raw(raw)
        return mdn_negative_log_likelihood(logits, mu, raw_sigma, y, min_sigma=self.min_sigma, max_sigma=self.max_sigma)

    def _eval_loss(self, net, X, y_tensor, task: str, device) -> float:
        net.eval()
        with torch.no_grad():
            xb = torch.as_tensor(X, dtype=torch.float32, device=device)
            yb = y_tensor.to(device)
            loss = self._training_loss(net(xb), yb, task)
        return float(loss.detach().cpu().item())

    def _sample_regression(self, column: str, H_context, rng, return_trace: bool):
        model = self.sampler_models_[column]
        logits, mu, sigma = self._predict_mdn_params(model, H_context)
        probabilities = self._softmax_np(logits / max(float(self.component_temperature), 1e-12))
        values = []
        traces = []
        for row_prob, row_mu, row_sigma in zip(probabilities, mu, sigma):
            p = row_prob / row_prob.sum() if row_prob.sum() else np.ones_like(row_prob) / len(row_prob)
            component = int(rng.choice(np.arange(len(p)), p=p))
            component_mu = float(row_mu[component])
            component_sigma = float(row_sigma[component])
            scaled_sigma = max(component_sigma * float(self.noise_scale), 1e-12)
            value = float(rng.normal(component_mu, scaled_sigma))
            values.append(value)
            if return_trace:
                log_density = self._mdn_log_density(value, row_prob, row_mu, row_sigma)
                traces.append(
                    {
                        "component_index": component,
                        "component_probability": float(p[component]),
                        "component_mean": component_mu,
                        "component_sigma": component_sigma,
                        "sample_log_probability": float(log_density),
                    }
                )
        values = np.asarray(values, dtype=float)
        return (values, traces) if return_trace else values

    def _sample_classification(self, column: str, H_context, rng, return_trace: bool):
        model = self.sampler_models_[column]
        proba = self._predict_class_proba(model, H_context)
        classes = np.asarray(model["classes"])
        values = []
        traces = []
        for row_proba in proba:
            p = np.asarray(row_proba, dtype=float)
            p = p / p.sum() if p.sum() else np.ones_like(p) / len(p)
            chosen_pos = int(rng.choice(np.arange(len(classes)), p=p))
            values.append(classes[chosen_pos])
            if return_trace:
                traces.append(
                    {
                        "sampled_class_code": int(classes[chosen_pos]) if np.issubdtype(classes.dtype, np.integer) else classes[chosen_pos],
                        "class_probabilities": {str(classes[i]): float(p[i]) for i in range(len(classes))},
                    }
                )
        values = np.asarray(values)
        return (values, traces) if return_trace else values

    def _predict_mdn_params(self, model, X):
        raw = self._predict_raw(model, X)
        logits, mu, raw_sigma = np.split(raw, 3, axis=1)
        sigma = np.log1p(np.exp(-np.abs(raw_sigma))) + np.maximum(raw_sigma, 0) + self.min_sigma
        sigma = np.minimum(sigma, self.max_sigma)
        return logits, mu, sigma

    def _predict_class_proba(self, model, X):
        raw = self._predict_raw(model, X)
        return self._softmax_np(raw)

    def _predict_raw(self, model, X):
        X_arr = self._as_float32_array(X)
        network = model["network"]
        device = model["device"]
        network.eval()
        with torch.no_grad():
            xb = torch.as_tensor(X_arr, dtype=torch.float32, device=device)
            return network(xb).detach().cpu().numpy()

    def _split_mdn_raw(self, raw):
        return torch.split(raw, self.n_components, dim=1)

    def _resolve_device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @staticmethod
    def _softmax_np(x):
        x = np.asarray(x, dtype=float)
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        denom = exp_x.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return exp_x / denom

    @staticmethod
    def _mdn_log_density(value, probabilities, mu, sigma):
        var = np.square(sigma)
        log_components = np.log(probabilities + 1e-300) - 0.5 * (np.log(2.0 * np.pi * var) + np.square(value - mu) / var)
        max_log = np.max(log_components)
        return max_log + np.log(np.exp(log_components - max_log).sum())

    @staticmethod
    def _as_float32_array(X):
        if sparse.issparse(X):
            return X.toarray().astype(np.float32, copy=False)
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    @staticmethod
    def _require_torch():
        if torch is None:
            raise ImportError("NeuralConditionalSampler requires torch")
