import torch
from sklearn.datasets import make_sparse_coded_signal
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from contextlib import contextmanager
from timeit import default_timer

from omp import omp_naive, omp_v0


n_components, n_features = 100, 100
n_nonzero_coefs = 17
n_samples = 50

@contextmanager
def elapsed_timer():
    # https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


def run_omp(X, y, n_nonzero_coefs, precompute=True, tol=0.0, normalize=False, fit_intercept=False, alg='naive'):
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X)
        y = torch.as_tensor(y)

    # TODO: Convert to pytorch tensor at this level?
    # We can either return sets, (sets, solutions), or xests
    # These are all equivalent, but are simply more and more dense representations.
    # Given sets and X and y one can (re-)construct xests. The second is just a sparse vector repr.

    # https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/linear_model/_omp.py#L690
    if fit_intercept or normalize:
        X = X.clone()
        assert not isinstance(precompute, torch.Tensor), "If user pre-computes XTX they can also pre-normalize X" \
                                                         " as well, so normalize and fit_intercept must be set false."

    if fit_intercept:
        X = X - X.mean(0)
        y = y - y.mean(1)[:, None]

    # To keep a good condition number on X, especially with Cholesky compared to LU factorization,
    # we should probably always normalize it (OMP is invariant anyways)
    if normalize is True:
        normalize = (X * X).sum(0).sqrt()  # User can also just optionally supply pre-computed norms.
        X /= normalize[None, :]  # Save compute if already normalized!

    if precompute is True or alg == 'v0':
        precompute = X.T @ X

    # If n_nonzero_coefs is equal to M, one should just return lstsq
    if alg == 'naive':
        sets, solutions, lengths = omp_naive(X, y, n_nonzero_coefs=n_nonzero_coefs, XTX=precompute, tol=tol)
    elif alg == 'v0':
        sets, solutions, lengths = omp_v0(X, y, n_nonzero_coefs=n_nonzero_coefs, XTX=precompute, tol=tol)

    solutions = solutions.squeeze(-1)
    if normalize is not False:
        solutions /= normalize[sets]

    xests = y.new_zeros(y.shape[0], X.shape[1])
    if lengths is None:
        xests[torch.arange(y.shape[0], dtype=sets.dtype, device=sets.device)[:, None], sets] = solutions
    else:
        for i in range(y.shape[0]):
            # print(sets.shape, xests[i, sets[i, :lengths[i]]].shape)
            # xests[i].scatter_(-1, sets[i, :lengths[i]], solutions[i, :lengths[i]])
            xests[i, sets[i, :lengths[i]]] = solutions[i, :lengths[i]]

    return xests


if __name__ == "__main__":
    # The naive algorithm has a memory complexity of kNM = O(N^2M), while the v0 has k(N^2+N(M+k)) = O(N^3+N^2M).
    # If k is modest
    # And all the other proposed algs will also

    # TODO: https://roman-kh.github.io/numpy-multicore/
    y, X, w = make_sparse_coded_signal(
        n_samples=n_samples,
        n_components=n_components,
        n_features=n_features,
        n_nonzero_coefs=n_nonzero_coefs,
        random_state=0)

    y = (y.T + np.random.randn(*y.T.shape) * 0.01)
    XTX = X.T @ X
    print("Settings used for the test: ")
    print("Number of Samples: " + str(n_samples))
    print("Number of Components: " + str(n_components))
    print("Number of Features: " + str(n_features))
    print("Number of Nonzero Coefficients: " + str(n_nonzero_coefs))
    print("\n")

    print('Single core. v0 fast implementation.')
    tol = 0.1
    k = 0
    with elapsed_timer() as elapsed:
        xests_v0 = run_omp(torch.as_tensor(X.copy()), torch.as_tensor(y.copy()), n_nonzero_coefs-k, normalize=True, fit_intercept=True, tol=tol, alg='v0')
    print('Samples per second:', n_samples / elapsed())
    print("\n")

    with elapsed_timer() as elapsed:
        xests_naive_fast = run_omp(X.copy().astype(np.float), y.copy().astype(np.float), n_nonzero_coefs-k, tol=tol, normalize=True, fit_intercept=True, alg='naive')
    print('Samples per second:', n_samples / elapsed())
    print("\n")
    print(xests_v0.numpy().nonzero(), xests_v0.shape)
    print('error in new code (v0)', np.max(np.abs(xests_v0.numpy() - xests_naive_fast.numpy())))

    if True:
        omp_args = dict(tol=tol, n_nonzero_coefs=n_nonzero_coefs-k, precompute=False, fit_intercept=True, normalize=True)
        # Single core
        print('Single core. Sklearn')
        omp = OrthogonalMatchingPursuit(**omp_args)
        with elapsed_timer() as elapsed:
            omp.fit(X, y.T)
        print('Samples per second:', n_samples / elapsed())
        print("\n")

        # print(omp.coef_[0].nonzero(), omp.coef_.shape, xests_naive_fast.shape, xests_naive_fast.dtype)
        # print(xests_naive_fast[0].numpy().nonzero())
        print((np.linalg.norm(y[..., None] - X @ omp.coef_[..., None], ord=2, axis=-2).squeeze(-1) ** 2).max())
        print((np.linalg.norm(y[..., None] - X @ xests_v0.numpy()[..., None], ord=2, axis=-2).squeeze(-1) ** 2).max())
        print((np.linalg.norm(y[..., None] - X @ xests_naive_fast.numpy()[..., None], ord=2, axis=-2).squeeze(-1) ** 2).max())
        print('error in new code (blas)', np.max(np.abs(omp.coef_ - xests_v0.numpy())))
        print('error in new code (blas)', np.max(np.abs(omp.coef_ - xests_naive_fast.numpy())))
        # print('idx in new code (blas)', np.max(np.abs(omp.coef_[5].nonzero()[0] - xests_naive_fast[5].numpy().nonzero()[0])))
        # print('idx in new code (blas)', np.max(np.abs(omp.coef_.nonzero()[1] - xests_naive_fast.numpy().nonzero()[1])))
