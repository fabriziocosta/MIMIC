"""Fast nearest-neighbour indexes for MIMIC embedding spaces."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class FastNeighborIndex:
    """Use IMSTE's approximate Euclidean backends for larger training sets.

    RINND is preferred, followed by PyNNDescent. Small datasets use exact
    scikit-learn search, where approximate index construction is not useful.
    """

    exact_search_max_samples = 128

    def __init__(self, X, *, n_neighbors: int, random_state: int | None):
        features = np.ascontiguousarray(X, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError("X must be a two-dimensional embedding matrix")
        if not np.isfinite(features).all():
            raise ValueError("Embeddings must contain only finite values")
        if len(features) < 2:
            raise ValueError("At least two embeddings are required")

        self.features_ = features
        self.n_samples_ = len(features)
        self.graph_width_ = min(max(1, int(n_neighbors)), self.n_samples_ - 1)
        self.query_index_ = None
        if self.n_samples_ > self.exact_search_max_samples:
            self.query_index_ = self._fit_rinnd(random_state)
            if self.query_index_ is None:
                self.query_index_ = self._fit_pynndescent(random_state)
        if self.query_index_ is None:
            self.backend_ = "sklearn_exact"
            self.query_index_ = NearestNeighbors(
                n_neighbors=min(self.graph_width_ + 1, self.n_samples_),
                metric="euclidean",
                n_jobs=-1,
            ).fit(features)
            distances, indices = self.query_index_.kneighbors(
                features, n_neighbors=min(self.graph_width_ + 1, self.n_samples_),
                return_distance=True,
            )
        else:
            distances, indices = self.query_index_.neighbor_graph
            self.backend_ = getattr(self, "backend_", "approximate")

        self.neighbor_indices_, self.neighbor_distances_ = self._clean_training_graph(
            indices, distances
        )

    def _fit_rinnd(self, random_state):
        try:
            from rinnd import RINND
        except ImportError:
            return None

        n = self.n_samples_
        width = min(self.graph_width_ + 1, n)
        options = {
            "metric": "euclidean",
            "n_neighbors": width,
            "random_state": random_state,
            "n_trees": max(4, min(8, 5 + int(round(n**0.5 / 20.0)))),
            "n_iters": max(5, int(round(np.log2(n)))),
            "max_candidates": max(20, min(60, int(1.5 * width))),
            "verbose": False,
        }
        try:
            options["n_jobs"] = -1
            index = RINND(self.features_, **options)
        except TypeError:
            options.pop("n_jobs", None)
            try:
                index = RINND(self.features_, **options)
            except (TypeError, ValueError, RuntimeError):
                return None
        except (ValueError, RuntimeError):
            return None
        self.backend_ = "rinnd"
        return index

    def _fit_pynndescent(self, random_state):
        try:
            from pynndescent import NNDescent
        except ImportError:
            return None
        width = min(self.graph_width_ + 1, self.n_samples_)
        try:
            index = NNDescent(
                self.features_,
                n_neighbors=width,
                metric="euclidean",
                random_state=random_state,
                n_jobs=-1,
            )
        except (TypeError, ValueError, RuntimeError):
            return None
        self.backend_ = "pynndescent"
        return index

    def _clean_training_graph(self, indices, distances):
        indices = np.asarray(indices, dtype=np.int64)
        distances = np.asarray(distances, dtype=np.float64)
        result_indices = np.full((self.n_samples_, self.graph_width_), -1, dtype=np.int64)
        result_distances = np.full((self.n_samples_, self.graph_width_), np.inf, dtype=np.float64)
        for row in range(self.n_samples_):
            valid = (
                (indices[row] >= 0)
                & (indices[row] < self.n_samples_)
                & (indices[row] != row)
                & np.isfinite(distances[row])
            )
            row_indices = indices[row, valid]
            row_distances = distances[row, valid]
            order = np.argsort(row_distances, kind="stable")[: self.graph_width_]
            count = len(order)
            result_indices[row, :count] = row_indices[order]
            result_distances[row, :count] = row_distances[order]
        return result_indices, result_distances

    def training_neighbors(self, row: int):
        indices = self.neighbor_indices_[int(row)]
        return indices[indices >= 0]

    def kneighbors(self, X, *, n_neighbors: int, return_distance: bool = True):
        query = np.ascontiguousarray(X, dtype=np.float32)
        k = min(max(1, int(n_neighbors)), self.n_samples_)
        if self.backend_ == "sklearn_exact":
            distances, indices = self.query_index_.kneighbors(
                query, n_neighbors=k, return_distance=True
            )
        else:
            indices, distances = self.query_index_.query(query, k=k)
            indices = np.asarray(indices, dtype=np.int64)
            distances = np.asarray(distances, dtype=np.float64)
            order = np.argsort(distances, axis=1, kind="stable")
            indices = np.take_along_axis(indices, order, axis=1)
            distances = np.take_along_axis(distances, order, axis=1)
        return (distances, indices) if return_distance else indices
