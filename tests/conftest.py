import numpy as np
import pytest


@pytest.fixture
def omp_problem():
    """Small well-conditioned OMP problem for fast tests.

    Returns (X, y_multi, y_single, coef_true, n_nonzero).
    X columns are normalized.  y = X @ coef_true.T (noiseless).
    """
    rng = np.random.RandomState(42)
    n_samples, n_features, n_targets = 50, 20, 5
    n_nonzero = 3

    X = rng.randn(n_samples, n_features)
    X /= np.linalg.norm(X, axis=0)  # normalize columns

    coef_true = np.zeros((n_targets, n_features))
    for t in range(n_targets):
        idx = rng.choice(n_features, n_nonzero, replace=False)
        coef_true[t, idx] = rng.randn(n_nonzero)

    y_multi = X @ coef_true.T  # (n_samples, n_targets)
    y_single = y_multi[:, 0]   # (n_samples,)

    return X, y_multi, y_single, coef_true, n_nonzero
