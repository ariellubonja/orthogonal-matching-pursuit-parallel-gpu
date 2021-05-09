import os
import torch
import torch.utils
import torch.utils.data
from sklearn.datasets import make_sparse_coded_signal
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from contextlib import contextmanager
from timeit import default_timer
from test_omp import omp_naive
from test import *  # FIXME: better name

n_components, n_features = 1024, 100
n_nonzero_coefs = 17
n_samples = 7000

@contextmanager
def elapsed_timer():
    # https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def run_omp(X, y, n_nonzero_coefs, XTX=None, normalize=False, fit_intercept=False, regularized=True):
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
        assert XTX is None or XTX is True  # in case user precomputes XTX they can also precompute this as well.

    if fit_intercept:
        X = X - X.mean(0)
        y = y - y.mean(0)

    # To keep a good condition number on X, especially with Cholesky compared to LU factorization,
    # we should probably always normalize it (OMP is invariant anyways)
    if normalize is True:
        normalize = (X * X).sum(0).sqrt()  # User can also just optionally supply pre-computed norms.
        X /= normalize[None, :]  # Save compute if already normalized!

    if XTX is True:
        XTX = X.T @ X

    # If n_nonzero_coefs is equal to M, one should just return lstsq
    sets, solutions = omp_naive_final(X, y, n_nonzero_coefs, XTX=XTX, ridge=regularized)
    if normalize is not False:
        solutions /= normalize[sets.t(), None]

    xests = y.new_zeros(y.shape[0], X.shape[1])
    xests[torch.arange(y.shape[0], dtype=sets.dtype, device=sets.device)[:, None], sets.t()] = solutions.squeeze(-1)

    return xests

def batch_mm(matrix, matrix_batch, return_contiguous=True):
    """
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    # One dgemm is faster than many dgemv.
    # From https://github.com/pytorch/pytorch/issues/14489#issuecomment-607730242
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose([1, 0, 2]).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    if return_contiguous:
        result = np.empty_like(matrix_batch, shape=(batch_size, matrix.shape[0], matrix_batch.shape[2]))
        np.matmul(matrix, vectors, out=result.transpose([1, 0, 2]).reshape(matrix.shape[0], -1))
    else:
        result = (matrix @ vectors).reshape(matrix.shape[0], batch_size, -1).transpose([1, 0, 2])

    return result


def innerp(x, y, out=None):
    if out is not None:
        out = out[:, None, None]  # Add space for two singleton dimensions.
    return torch.matmul(x[..., None, :], y[..., :, None], out=out)[..., 0, 0]

def cholesky_solve(ATA, ATy):
    if ATA.dtype == torch.half or ATy.dtype == torch.half:
        ATA = ATA.to(torch.float)
        ATy = ATy.to(torch.float)
    return ATy.cholesky_solve(torch.cholesky(ATA)).to(ATy.dtype)


def omp_naive_final(X, y, n_nonzero_coefs, tol=0.5, XTX=None, ridge=None):
    tol = tol * tol
    on_cpu = not y.is_cuda and not y.dtype == torch.half
    ridge = None  # FIXME: kill this
    # Given X as an MxN array and y as an BxN array, do omp to approximately solve Xb=y

    # Base variables
    XT = X.contiguous().t()  # Store XT in fortran-order.
    y = y.contiguous()
    r = y.clone()
    sets = y.new_zeros((n_nonzero_coefs, y.shape[0]), dtype=torch.long).t()
    original_indices = torch.arange(y.shape[0], dtype=sets.dtype, device=sets.device)
    new_indices = torch.arange(y.shape[0], dtype=sets.dtype, device=sets.device)
    n_fin = sets.new_zeros((1,))

    # Trade b*k^2+bk+bkM = O(bkM) memory for much less compute time. (This has to be done anyways since we are batching,
    # otherwise one could just permute columns of X in-place as in https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/linear_model/_omp.py#L28 )
    ATs = y.new_zeros(r.shape[0], n_nonzero_coefs, X.shape[0])
    ATys = y.new_zeros(r.shape[0], n_nonzero_coefs, 1)
    ATAs = torch.eye(n_nonzero_coefs, dtype=y.dtype, device=y.device)[None].repeat(r.shape[0], 1, 1)
    if on_cpu:
        # For CPU it is faster to use a packed representation of the lower triangle in ATA.
        tri_idx = torch.tril_indices(n_nonzero_coefs, n_nonzero_coefs, device=sets.device, dtype=sets.dtype)
        ATAs = ATAs[:, tri_idx[0], tri_idx[1]]

    for k in range(n_nonzero_coefs):
        # STOPPING CRITERIA
        if tol:
            normr2 = innerp(r, r)
            problems_done = normr2 < tol
            if problems_done.any():
                remaining = ~problems_done
                new_fin = problems_done.sum()
                # vals, idxs = torch.sort(normr2)
                # new_indices[n_fin:n_fin + new_fin] = original_indices[idxs[:new_fin]]  # [n_fin:]
                new_indices[n_fin:n_fin + new_fin] = original_indices[problems_done]  # [n_fin:]
                original_indices = original_indices[remaining]
                ATs = ATs[remaining]
                ATys = ATys[remaining]
                ATAs = ATAs[remaining]
                sets = sets[remaining]
                y = y[remaining]
                r = r[remaining]


        # GET PROJECTIONS AND INDICES TO ADD
        if on_cpu:
            projections = batch_mm(XT.numpy(), r[:, :, None].numpy())
            argmax_blast(projections.squeeze(-1), sets[:, k].numpy())
        else:
            projections = XT @ r[:, :, None]
            sets[:, k] = projections.abs().sum(-1).argmax(-1)  # Sum is just a squeeze, but would be relevant in SOMP.

        # UPDATE AT
        AT = ATs[:, :k + 1, :]
        updateA = XT[sets[:, k], :]
        AT[:, k, :] = updateA

        # UPDATE ATy based on AT
        ATy = ATys[:, :k + 1]
        innerp(updateA, y, out=ATy[:, k, 0])

        # UPDATE ATA based on AT or XTX.
        if on_cpu:
            packed_idx = k * (k - 1) // 2
            if XTX is not None:  # Update based on precomputed XTX.
                ATAs[k + packed_idx:packed_idx + 2 * k + 1, :].t().numpy()[:] = XTX[sets[:, k, None], sets[:, :k + 1]]
            else:
                np.matmul(AT[:, :k + 1, :].numpy(), updateA[:, :, None].numpy(),
                          out=ATAs.t()[k + packed_idx:packed_idx + 2 * k + 1, :].t()[:, :, None].numpy())
        else:
            if XTX is not None:
                ATA[:, k, :k + 1] = XTX[sets[:, k, None], sets[:, :k + 1]]
            else:
                # Update ATAs by adding the new column of inner products.
                ATA = ATAs[:, :k + 1, :k + 1]  # (or we'll be updating the packed version of it)
                torch.bmm(AT[:, :k + 1, :], updateA[:, :, None], out=ATA[:, k, :k + 1, None])

        # SOLVE ATAx = ATy.
        if on_cpu:
            solutions = ATy.permute(0, 2, 1).clone().permute(0, 2, 1)  # Get a copy.
            ppsv(ATAs.t()[:packed_idx + 2 * k + 1, :].t().contiguous().numpy(), solutions.numpy())
        else:
            ATA[:, :k, k] = ATA[:, k, :k]  # Copy lower triangle to upper triangle.
            solutions = cholesky_solve(ATA, ATy)

        # FINALLY, GET NEW RESIDUAL r=y-Ax
        if on_cpu:
            np.subtract(y.numpy(), (AT.permute(0, 2, 1).numpy() @ solutions.numpy()).squeeze(-1), out=r.numpy())
        else:
            torch.baddbmm(y[:, :, None], AT.permute(0, 2, 1), solutions, beta=-1, out=r[:, :, None])

    return sets.t(), solutions


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

    y = y.T.copy()
    XTX = X.T @ X
    print("Settings used for the test: ")
    print("Number of Samples: " + str(n_samples))
    print("Number of Components: " + str(n_components))
    print("Number of Features: " + str(n_features))
    print("Number of Nonzero Coefficients: " + str(n_nonzero_coefs))
    print("\n")

    print('Single core. Naive fast implementation.')
    with elapsed_timer() as elapsed:
        xests_naive_fast = run_omp(X.copy(), y.copy(), n_nonzero_coefs)
    print('Samples per second:', n_samples / elapsed())
    print("\n")

    with elapsed_timer() as elapsed:
        xests_naive_fast = run_omp(X.copy(), y.copy(), n_nonzero_coefs)
    print('Samples per second:', n_samples / elapsed())
    print("\n")

    with elapsed_timer() as elapsed:
        xests_naive_fast = run_omp(X.copy(), y.copy(), n_nonzero_coefs)
    print('Samples per second:', n_samples / elapsed())
    print("\n")

    print('Single core. Naive implementation.')
    with elapsed_timer() as elapsed:
        xests_naive = omp_naive(X.copy(), y.T.copy(), n_nonzero_coefs)
    print('Samples per second:', n_samples / elapsed())
    print("\n")


    print('error in new code (blas)', np.max(np.abs(xests_naive - xests_naive_fast.numpy())))
