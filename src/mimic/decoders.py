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
