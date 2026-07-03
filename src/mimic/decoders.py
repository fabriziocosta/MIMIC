"""Decoder implementations for MIMIC."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.validation import check_is_fitted


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
