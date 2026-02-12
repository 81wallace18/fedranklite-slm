import numpy as np
import pytest

from src.data import _partition_indices


class TestIID:
    def test_all_clients_get_data(self):
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10)
        indices = _partition_indices(labels, n_clients=5, method="iid", alpha=1.0, min_samples=1, seed=42)
        assert len(indices) == 5
        for idx in indices:
            assert len(idx) > 0

    def test_covers_all_samples(self):
        labels = np.arange(100)
        indices = _partition_indices(labels, n_clients=10, method="iid", alpha=1.0, min_samples=1, seed=42)
        all_idx = np.concatenate(indices)
        assert len(all_idx) == 100
        assert len(set(all_idx)) == 100


class TestDirichlet:
    def test_all_clients_get_data(self):
        labels = np.array([0] * 50 + [1] * 50)
        indices = _partition_indices(labels, n_clients=5, method="dirichlet", alpha=0.1, min_samples=5, seed=42)
        assert len(indices) == 5
        for idx in indices:
            assert len(idx) >= 5

    def test_low_alpha_creates_skew(self):
        labels = np.array([0] * 200 + [1] * 200)
        indices = _partition_indices(labels, n_clients=4, method="dirichlet", alpha=0.01, min_samples=5, seed=42)
        # with very low alpha, each client should be skewed toward one label
        for idx in indices:
            client_labels = labels[idx]
            counts = np.bincount(client_labels, minlength=2)
            ratio = counts.max() / max(counts.sum(), 1)
            # at least 60% of one label (relaxed threshold for randomness)
            assert ratio > 0.6

    def test_high_alpha_approaches_iid(self):
        labels = np.array([0] * 200 + [1] * 200)
        indices = _partition_indices(labels, n_clients=4, method="dirichlet", alpha=100.0, min_samples=5, seed=42)
        for idx in indices:
            client_labels = labels[idx]
            counts = np.bincount(client_labels, minlength=2)
            ratio = counts.min() / max(counts.max(), 1)
            # should be roughly balanced
            assert ratio > 0.3

    def test_min_samples_enforced(self):
        labels = np.array([0] * 100 + [1] * 100)
        indices = _partition_indices(labels, n_clients=10, method="dirichlet", alpha=0.01, min_samples=10, seed=42)
        for idx in indices:
            assert len(idx) >= 10


class TestLabelSkew:
    def test_all_clients_get_data(self):
        labels = np.array([0] * 50 + [1] * 50 + [2] * 50 + [3] * 50)
        indices = _partition_indices(labels, n_clients=4, method="label_skew", alpha=1.0, min_samples=1, seed=42)
        assert len(indices) == 4
        for idx in indices:
            assert len(idx) > 0
