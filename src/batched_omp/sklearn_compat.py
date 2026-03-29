import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from .omp import run_omp


class BatchedOrthogonalMatchingPursuit(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """GPU-accelerated Orthogonal Matching Pursuit — drop-in replacement for
    ``sklearn.linear_model.OrthogonalMatchingPursuit``.

    Works with ``Pipeline``, ``GridSearchCV``, ``cross_val_score``, and every
    sklearn workflow.  Pass ``device="auto"`` (the default) to use CUDA when
    available or fall back to CPU transparently.

    Parameters
    ----------
    n_nonzero_coefs : int or None, default=None
        Maximum number of non-zero coefficients.  When *None* and *tol* is
        also *None*, defaults to ``max(int(0.1 * n_features), 1)``.
    tol : float or None, default=None
        Maximum **squared** residual norm.  Overrides *n_nonzero_coefs*.
    fit_intercept : bool, default=True
        Whether to center X and y before fitting.
    precompute : bool, default=True
        Whether to precompute the Gram matrix ``X.T @ X``.
    normalize : bool, default=True
        Normalize dictionary columns before OMP (improves numerical
        stability).  Coefficients are de-normalized on output.
    device : str, default="auto"
        ``"auto"`` selects CUDA if available, otherwise CPU.  Also accepts
        ``"cpu"``, ``"cuda"``, or any valid ``torch.device`` string.
    algorithm : str, default="v0"
        OMP algorithm variant: ``"v0"`` (inverse Cholesky, fastest),
        ``"v0_blas"`` (NumPy + Cython BLAS), or ``"naive"``.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Estimated sparse coefficients.
    intercept_ : float or ndarray of shape (n_targets,)
        Intercept (bias) term.
    n_iter_ : int or list of int
        Number of active features (OMP iterations) per target.
    n_nonzero_coefs_ : int
        Resolved value of *n_nonzero_coefs* used during fitting.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    def __init__(
        self,
        n_nonzero_coefs=None,
        tol=None,
        fit_intercept=True,
        precompute=True,
        normalize=True,
        device="auto",
        algorithm="v0",
    ):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.normalize = normalize
        self.device = device
        self.algorithm = algorithm

    def _resolve_device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def fit(self, X, y):
        import torch

        X, y = validate_data(
            self, X, y, multi_output=True, y_numeric=True, dtype="float64"
        )

        n_features = X.shape[1]

        # Resolve n_nonzero_coefs (sklearn default: 10% of features).
        if self.n_nonzero_coefs is None and self.tol is None:
            self.n_nonzero_coefs_ = max(int(0.1 * n_features), 1)
        elif self.tol is not None:
            self.n_nonzero_coefs_ = n_features  # upper bound; tol controls stopping
        else:
            self.n_nonzero_coefs_ = self.n_nonzero_coefs

        # Center ourselves so we can extract the intercept.
        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean(axis=0) if y.ndim > 1 else y.mean()
            X_c = X - X_mean
            y_c = y - y_mean
        else:
            X_c, y_c = X, y

        # run_omp expects y as (batch, signal_dim) — transpose from sklearn's
        # (n_samples, n_targets) convention.
        single_target = y.ndim == 1
        y_omp = y_c[np.newaxis, :] if single_target else y_c.T

        device = self._resolve_device()
        X_t = torch.as_tensor(X_c, device=device)
        y_t = torch.as_tensor(y_omp, device=device)

        coef = run_omp(
            X_t,
            y_t,
            n_nonzero_coefs=self.n_nonzero_coefs_,
            precompute=self.precompute,
            tol=self.tol if self.tol is not None else 0.0,
            normalize=self.normalize,
            fit_intercept=False,  # we already centered
            alg=self.algorithm,
        )

        self.coef_ = coef.cpu().numpy()  # (n_targets, n_features)
        if single_target:
            self.coef_ = self.coef_.squeeze(0)  # (n_features,)

        # Intercept: y_mean - X_mean @ coef
        if self.fit_intercept:
            if single_target:
                self.intercept_ = float(y_mean - X_mean @ self.coef_)
            else:
                self.intercept_ = y_mean - X_mean @ self.coef_.T
        else:
            self.intercept_ = 0.0

        # n_iter_ = number of non-zero coefficients per target (= OMP iterations).
        if single_target:
            self.n_iter_ = int(np.count_nonzero(self.coef_))
        else:
            self.n_iter_ = np.count_nonzero(self.coef_, axis=1).tolist()

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        if self.coef_.ndim == 1:
            return X @ self.coef_ + self.intercept_
        return X @ self.coef_.T + self.intercept_
