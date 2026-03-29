import numpy as np
from contextlib import contextmanager
from timeit import default_timer

import torch


@contextmanager
def elapsed_timer():
    # https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


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
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k).
    if return_contiguous:
        result = np.empty_like(matrix_batch, shape=(batch_size, matrix.shape[0], matrix_batch.shape[2]))
        np.matmul(matrix, vectors, out=result.transpose([1, 0, 2]).reshape(matrix.shape[0], -1))
    else:
        result = (matrix @ vectors).reshape(matrix.shape[0], batch_size, -1).transpose([1, 0, 2])

    return result


def innerp(x, y=None, out=None):
    if y is None:
        y = x
    if out is not None:
        out = out[:, None, None]  # Add space for two singleton dimensions.
    return torch.matmul(x[..., None, :], y[..., :, None], out=out)[..., 0, 0]


def cholesky_solve(ATA, ATy):
    if ATA.dtype == torch.half or ATy.dtype == torch.half:
        return ATy.to(torch.float).cholesky_solve(torch.linalg.cholesky(ATA.to(torch.float))).to(ATy.dtype)
    return ATy.cholesky_solve(torch.linalg.cholesky(ATA)).to(ATy.dtype)
