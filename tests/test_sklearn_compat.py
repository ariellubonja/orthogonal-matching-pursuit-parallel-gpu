import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from batched_omp import BatchedOrthogonalMatchingPursuit


# ---------------------------------------------------------------------------
# Basic fit / predict
# ---------------------------------------------------------------------------

class TestSingleTarget:
    def test_coef_shape(self, omp_problem):
        X, _, y, _, n_nonzero = omp_problem
        est = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        est.fit(X, y)
        assert est.coef_.shape == (X.shape[1],)

    def test_intercept_is_scalar(self, omp_problem):
        X, _, y, _, n_nonzero = omp_problem
        est = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        est.fit(X, y)
        assert np.isscalar(est.intercept_)

    def test_predict_shape(self, omp_problem):
        X, _, y, _, n_nonzero = omp_problem
        est = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        est.fit(X, y)
        pred = est.predict(X)
        assert pred.shape == y.shape

    def test_n_iter(self, omp_problem):
        X, _, y, _, n_nonzero = omp_problem
        est = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        est.fit(X, y)
        assert isinstance(est.n_iter_, int)
        assert 1 <= est.n_iter_ <= n_nonzero


class TestMultiTarget:
    def test_coef_shape(self, omp_problem):
        X, y, _, _, n_nonzero = omp_problem
        n_targets = y.shape[1]
        est = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        est.fit(X, y)
        assert est.coef_.shape == (n_targets, X.shape[1])

    def test_intercept_shape(self, omp_problem):
        X, y, _, _, n_nonzero = omp_problem
        est = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        est.fit(X, y)
        assert est.intercept_.shape == (y.shape[1],)

    def test_predict_shape(self, omp_problem):
        X, y, _, _, n_nonzero = omp_problem
        est = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        est.fit(X, y)
        pred = est.predict(X)
        assert pred.shape == y.shape

    def test_n_iter_is_list(self, omp_problem):
        X, y, _, _, n_nonzero = omp_problem
        est = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        est.fit(X, y)
        assert isinstance(est.n_iter_, list)
        assert len(est.n_iter_) == y.shape[1]


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestCorrectness:
    def test_reconstruction(self, omp_problem):
        """Noiseless problem: reconstruction error should be near zero."""
        X, _, y, _, n_nonzero = omp_problem
        est = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        est.fit(X, y)
        pred = est.predict(X)
        rel_err = np.linalg.norm(pred - y) / np.linalg.norm(y)
        assert rel_err < 1e-6, f"Relative reconstruction error too high: {rel_err}"

    def test_sparsity(self, omp_problem):
        """Coefficients should have at most n_nonzero_coefs non-zeros."""
        X, _, y, _, n_nonzero = omp_problem
        est = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        est.fit(X, y)
        assert np.count_nonzero(est.coef_) <= n_nonzero

    def test_multi_target_reconstruction(self, omp_problem):
        X, y, _, _, n_nonzero = omp_problem
        est = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        est.fit(X, y)
        pred = est.predict(X)
        for t in range(y.shape[1]):
            rel_err = np.linalg.norm(pred[:, t] - y[:, t]) / np.linalg.norm(y[:, t])
            assert rel_err < 1e-6, f"Target {t}: rel error {rel_err}"


# ---------------------------------------------------------------------------
# Drop-in replacement for sklearn
# ---------------------------------------------------------------------------

class TestDropIn:
    def test_same_coef_shape_single(self, omp_problem):
        X, _, y, _, n_nonzero = omp_problem
        sk = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        ours = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        sk.fit(X, y)
        ours.fit(X, y)
        assert sk.coef_.shape == ours.coef_.shape

    def test_same_coef_shape_multi(self, omp_problem):
        X, y, _, _, n_nonzero = omp_problem
        sk = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        ours = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        sk.fit(X, y)
        ours.fit(X, y)
        assert sk.coef_.shape == ours.coef_.shape

    def test_reconstruction_at_least_as_good(self, omp_problem):
        """Our reconstruction error should be <= sklearn's (or very close)."""
        X, _, y, _, n_nonzero = omp_problem
        sk = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        ours = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        sk.fit(X, y)
        ours.fit(X, y)
        sk_err = np.linalg.norm(sk.predict(X) - y)
        our_err = np.linalg.norm(ours.predict(X) - y)
        # Allow 10% tolerance — algorithms may pick different atoms.
        assert our_err <= sk_err * 1.1 + 1e-10, (
            f"Our error {our_err:.6e} > sklearn's {sk_err:.6e}"
        )


# ---------------------------------------------------------------------------
# sklearn integration
# ---------------------------------------------------------------------------

class TestSklearnIntegration:
    def test_pipeline(self, omp_problem):
        X, _, y, _, n_nonzero = omp_problem
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("omp", BatchedOrthogonalMatchingPursuit(
                n_nonzero_coefs=n_nonzero, normalize=False)),
        ])
        pipe.fit(X, y)
        pred = pipe.predict(X)
        assert pred.shape == y.shape

    def test_gridsearch(self, omp_problem):
        X, _, y, _, _ = omp_problem
        est = BatchedOrthogonalMatchingPursuit()
        gs = GridSearchCV(est, {"n_nonzero_coefs": [2, 3, 5]}, cv=3)
        gs.fit(X, y)
        assert hasattr(gs, "best_params_")
        assert gs.best_params_["n_nonzero_coefs"] in [2, 3, 5]

    def test_clone(self, omp_problem):
        est = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=5, device="cpu")
        est2 = clone(est)
        assert est2.n_nonzero_coefs == 5
        assert est2.device == "cpu"
        assert not hasattr(est2, "coef_")  # clone should not copy fitted state


# ---------------------------------------------------------------------------
# Parameter edge cases
# ---------------------------------------------------------------------------

class TestParameters:
    def test_no_fit_intercept(self, omp_problem):
        X, _, y, _, n_nonzero = omp_problem
        est = BatchedOrthogonalMatchingPursuit(
            n_nonzero_coefs=n_nonzero, fit_intercept=False)
        est.fit(X, y)
        assert est.intercept_ == 0.0

    def test_default_n_nonzero_coefs(self, omp_problem):
        X, _, y, _, _ = omp_problem
        est = BatchedOrthogonalMatchingPursuit()
        est.fit(X, y)
        expected = max(int(0.1 * X.shape[1]), 1)
        assert est.n_nonzero_coefs_ == expected

    def test_device_cpu(self, omp_problem):
        X, _, y, _, n_nonzero = omp_problem
        est = BatchedOrthogonalMatchingPursuit(
            n_nonzero_coefs=n_nonzero, device="cpu")
        est.fit(X, y)
        assert est.coef_.shape == (X.shape[1],)

    def test_n_features_in(self, omp_problem):
        X, _, y, _, n_nonzero = omp_problem
        est = BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        est.fit(X, y)
        assert est.n_features_in_ == X.shape[1]


# ---------------------------------------------------------------------------
# GPU tests (skipped if no CUDA)
# ---------------------------------------------------------------------------

def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


gpu = pytest.mark.skipif(
    not _has_cuda(), reason="No CUDA GPU available"
)


@gpu
class TestGPU:
    def test_gpu_matches_cpu(self, omp_problem):
        X, _, y, _, n_nonzero = omp_problem
        cpu = BatchedOrthogonalMatchingPursuit(
            n_nonzero_coefs=n_nonzero, device="cpu")
        cuda = BatchedOrthogonalMatchingPursuit(
            n_nonzero_coefs=n_nonzero, device="cuda")
        cpu.fit(X, y)
        cuda.fit(X, y)
        np.testing.assert_allclose(cpu.coef_, cuda.coef_, atol=1e-10)


# ---------------------------------------------------------------------------
# sklearn estimator contract (parametrize_with_checks)
# ---------------------------------------------------------------------------

from sklearn.utils.estimator_checks import parametrize_with_checks


@parametrize_with_checks([
    BatchedOrthogonalMatchingPursuit(n_nonzero_coefs=2),
])
def test_sklearn_estimator_contract(estimator, check):
    check(estimator)
