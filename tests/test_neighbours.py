import numpy as np

from mimic.neighbours import FastNeighborIndex


def test_fast_neighbor_index_uses_exact_search_for_small_data():
    X = np.random.default_rng(5).normal(size=(24, 4))
    index = FastNeighborIndex(X, n_neighbors=5, random_state=3)

    assert index.backend_ == "sklearn_exact"
    assert len(index.training_neighbors(0)) == 5
    assert 0 not in index.training_neighbors(0)
    distances, indices = index.kneighbors(X[:3], n_neighbors=4)
    assert distances.shape == indices.shape == (3, 4)
    assert np.all(np.diff(distances, axis=1) >= 0)


def test_fast_neighbor_index_uses_approximate_backend_for_larger_data():
    X = np.random.default_rng(8).normal(size=(160, 6))
    index = FastNeighborIndex(X, n_neighbors=6, random_state=11)

    assert index.backend_ in {"rinnd", "pynndescent"}
    assert len(index.training_neighbors(0)) == 6
    assert 0 not in index.training_neighbors(0)
    distances, indices = index.kneighbors(X[:2], n_neighbors=5)
    assert distances.shape == indices.shape == (2, 5)
    assert np.all(np.diff(distances, axis=1) >= 0)
