import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal
from scipy.io import loadmat
import random
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
from tqdm import tqdm
import os
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import *
from torch.utils.data.sampler import *
from line_profiler import line_profiler

from contextlib import contextmanager
from timeit import default_timer

# cholesky decomposition has a complexity of O(n^3) - the same as simply solving an equation.
#  then solving an equation using a cholesky (or LU, or another triangular form), has complexity O(n^2)
#  so I think there is only really a benefit if we have many LHS's.
# one can use memoization - e.g. for N=64 columns there will be about 1/64 probability of a hit on the first few iters
#  (k is low, << 64). -- We can conclude that in general it probably wont be worth the effort (though in cases of
#   non-random sensing matrices and non-random data, it may be worth, as many may take the same path.)
#  on iteration 2, the probability of a hit becomes 1/64^2..., one can then e.g. use matrix-inverse-updates, so they
#  don't have to match perfectly (bringing the probability back to ~1/64) - but still for any significant N, this is
#   just not going to give any significant speedups.


# We have MP, where we just use projection and select best result: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.64.5790&rep=rep1&type=pdf
# In OMP we do the same, but then pseudoinverse/least squares to get our representation.
# After we are done: Simultaneous orthogonal matching pursuit. [ https://arxiv.org/pdf/1506.05324.pdf ] We can actually implement this already now, all the times we insert [..., None] should just be replaced by the input already having a singleton dimension - then e.g. after projection we do a sum of absolutes of these values [*asum] before doing idamax! Everything else will work as-is in this case :D, oh, except the projections, where in this case we must do the matmul trick on every simultaneussy subset.

# FIXME: In case the dictonary is orthonormal, (O)MP is a kind of greedy PCA - but usually we have a very redundant dictonary, so the columns are not orthonormal (the rows are!) - our algorithms use that the columns are normalized, so rows are not necessarily orthonormal anymore?? (This seems like an issue! but maybe it is not since this normalization may cancel in pinv??? No...)
#        It seems since we normalize columns, we should have to take this into account somewhere? Why do we normalize columns again?...
#        Looking at the efficient implementation paper it seems this must be done - but this is exactly what column normalization does - but then we dont also want column normalization at the pseudo-inverse!?
# That the naive algorithm uses O(MN) memory seems to be an error - it seems this way because MATLAB hides the fact that to do pseudoinverse you need k^2 memory! (Of course peers reviewers missed this as well, and the mistake persists, even caries over to subsequent papers, now when they can cite a source...) even some programming wierdness such as maxmag/alpha non-use and index(1) persists to v0. So odd.
# From slides: OMP is better for a decaying signal - which is often the case for natural signals/(academia?) - and binary sparse signals something something industry?

# TODO: Use get_lapack_funcs to select appropriate and fastest functionsn (and possibly take into account work-memory) https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.get_lapack_funcs.html#scipy.linalg.lapack.get_lapack_funcs

# cublas cython: https://github.com/cupy/cupy/blob/4469fae998df33c72ff40ef954cb08b8f0004b18/cupy_backends/cuda/libs/cublas.pyx
# or cupy (custom kernels) + pytorch: https://tim37021.medium.com/pycuda-cupy-pytorch-interlop-56a9875e2aa7

@contextmanager
def elapsed_timer():
    # https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

if False:
    mat = loadmat('ps1_2021')
    Af = mat['Af']
    Ar = mat['Ar']
    yf = mat['yf']
    yr = mat['yr']
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=3)
    # X = np.random.randn(Af.shape[0], Af.shape[1])
    # X /= np.sqrt(np.sum((X ** 2), axis=0))
    omp.fit(Af, yf)
    print(omp.coef_)
    omp.fit(Ar, yr)
    print(omp.coef_)

# We can take a single solver, and multiple problems, and spread it out on processing units.
# ^ This can be done simply with the multiprocessing Python module.

n_components, n_features = 1024, 100
n_nonzero_coefs = 17
n_samples = 800
# For a setting like the one at https://sparse-plex.readthedocs.io/en/latest/book/pursuit/omp/fast_omp.html
#  it seems the naive algorithm is way fast when n_nonzero_coefs is small.

def solveomp(y):
    solveomp.omp.fit(solveomp.X, y)
    return solveomp.omp.coef_

def init_threads(func, X, omp_args):
    func.omp = OrthogonalMatchingPursuit(**omp_args)
    func.X = X

# TODO: Compare different matrix solving algorithms for small vs large matrices.
#       If it is large posv should always be faster. But if it is small, maybe gesv is better? (Or maybe posv already takes this into account?!)

# kernprof -l -v test_omp.py and @profile
# Based on https://github.com/zhuhufei/OMP/blob/master/codeAug2020.m
# From "Efficient Implementations for Orthogonal Matching Pursuit" (2020)
def omp_v0(y, X, XTX, XTy, n_nonzero_coefs=None):
    if n_nonzero_coefs is None:
        n_nonzero_coefs = X.shape[1]

    N = y.shape[1]
    innerp = lambda x: np.einsum('ij,ij->i', x, x)
    normr2 = innerp(y.T)  # Norm squared of residual.
    projections = XTy
    # Maybe do asfortranarray on XTX?

    gamma = np.zeros(shape=(n_nonzero_coefs, N), dtype=np.int64)
    F = np.repeat(np.identity(n_nonzero_coefs, dtype=X.dtype)[np.newaxis], N, 0)
    a_F = np.zeros_like(X, shape=(n_nonzero_coefs, N, 1))
    D_mybest = np.zeros_like(X, shape=(N, XTX.shape[0], n_nonzero_coefs))
    temp_F_k_k = 1
    xests = np.zeros((N, X.shape[1]))
    for k in range(n_nonzero_coefs):
        maxindices = np.argmax(projections * projections, 1)  # Maybe replace square with abs?
        gamma[k] = maxindices
        if k == 0:
            new_D_mybest = XTX[maxindices, :]
        else:
            D_mybest_maxindices = np.take_along_axis(D_mybest[:, :, :k], maxindices[:, None, None], 1).squeeze(1)
            temp_F_k_k = np.sqrt(1/(1-innerp(D_mybest_maxindices)))[:, None]
            # VVV In BLAS this can be calculated with *trmv  VVV
            F[:, :, k] = -temp_F_k_k * (F[:, :, :k] @ D_mybest_maxindices[:, :, None]).squeeze(-1)
            F[:, k, k] = temp_F_k_k[..., 0]
            new_D_mybest = temp_F_k_k * (XTX[maxindices, :] - (D_mybest[:, :, :k] @ D_mybest_maxindices[:, :, None]).squeeze(-1))
        D_mybest[:, :, k] = new_D_mybest
        a_F[k] = temp_F_k_k * np.take_along_axis(projections, maxindices[:, None], 1)  # TODO: Figure out if we should use XTy.T or XTy, which is faster?
        projections = projections - D_mybest[:, :, k] * a_F[k]
        normr2 = normr2 - (a_F[k] * a_F[k]).squeeze(-1)
    else:
        np.put_along_axis(xests, gamma.T, (F @ a_F.squeeze(-1).T[:, :, None]).squeeze(-1), -1)
        # Instead of putting a stopping criteria, we could return the order in which the coeffs were added. (i.e. just return gamma and amplitudes)
    # F[:, :k, :k] @ a_F.T[:k, :, None]
    return xests

from numba import jit, njit
@jit(nopython=True)
def get_max_projections(projections):
    # It seems BLAS ISAMAX/IDMAX should cover this? (Good for CUDA especially)
    B = projections.shape[0]
    result = np.zeros((B,), dtype=np.int64)
    for i in range(B):
        result[i] = np.argmax(np.abs(projections[i]))
    return result

from test import *
import scipy

def get_max_projections_blas(projections):
    #  BLAS is even faster! :O
    # Maybe remove overhead with Cython? https://stackoverflow.com/questions/44710838/calling-blas-lapack-directly-using-the-scipy-interface-and-cython, https://yiyibooks.cn/sorakunnn/scipy-1.0.0/scipy-1.0.0/linalg.cython_blas.html
    if False:
        func = 'i' + scipy.linalg.blas.find_best_blas_type(dtype=projections.dtype)[0] + 'amax'
        func = getattr(scipy.linalg.blas, func)
        B = projections.shape[0]
        result = np.zeros((B,), dtype=np.int64)
        for i in range(B):  # This has around a 20% Python overhead.
            result[i] = func(projections[i])  # np.argmax(np.abs(projections[i]))
        return result
    else:
        # print(projections.strides)
        # result2 = np.empty((projections.shape[0],), dtype=np.int64)
        result3 = np.empty((projections.shape[0],), dtype=np.int64)
        # print(projections.shape, result2.shape)
        # argmax_blas(projections, result2)  # seems to be about 10 times faster than the above
        argmax_blast(projections, result3)
        return result3


def update_projections_blas(projections, D_mybest, coefs):
    func = scipy.linalg.blas.find_best_blas_type(dtype=projections.dtype)[0] + 'axpy'
    func = getattr(scipy.linalg.blas, func)
    for i in range(projections.shape[0]):
        func(D_mybest[i], projections[i], a=coefs[i])  # np.argmax(np.abs(projections[i]))

@njit
def update_D_mybest(temp_F_k_k, XTX, maxindices, einssum):
    # We can possibly do this with BLAS (We will need BLAS to do gemv with alpha=-temp_F_k_k as the einsum,
    #  then followed by this here saxpy, with alpha=temp_F_k_k.)
    for i in range(temp_F_k_k.shape[0]):
         einssum[i] = temp_F_k_k[i] * (XTX[maxindices[i]] - einssum[i])

import time
def omp_v0_new(y, X, XTX, XTy, n_nonzero_coefs=None):
    if n_nonzero_coefs is None:
        n_nonzero_coefs = X.shape[1]

    B = y.shape[1]
    innerp = lambda x: np.einsum('ij,ij->i', x, x)
    normr2 = innerp(y.T)  # Norm squared of residual.
    projections = XTy

    gamma = np.zeros(shape=(n_nonzero_coefs, B), dtype=np.int64)
    F = np.repeat(np.identity(n_nonzero_coefs, dtype=X.dtype)[np.newaxis], B, 0)
    a_F = np.zeros_like(X, shape=(n_nonzero_coefs, B, 1))
    D_mybest = np.empty_like(X, shape=(n_nonzero_coefs, B, XTX.shape[1]))  # empty_like is faster to init
    temp_F_k_k = 1
    xests = np.zeros((B, X.shape[1]))
    for k in range(n_nonzero_coefs):
        maxindices = get_max_projections_blas(projections)  # Numba version is about twice as fast as: np.argmax(projections * projections, 1)  # Maybe replace square with abs?
        gamma[k] = maxindices
        if k == 0:
            D_mybest[k] = XTX[None, maxindices, :]
        else:
            # Do something about this:
            D_mybest_maxindices = np.take_along_axis(D_mybest[:k, :, :].transpose([2, 1, 0]), maxindices[None, :, None], 0).squeeze(0)
            temp_F_k_k = np.sqrt(1/(1-innerp(D_mybest_maxindices)))[:, None]
            F[:, :, k] = -temp_F_k_k * (F[:, :, :k] @ D_mybest_maxindices[:, :, None]).squeeze(-1)
            F[:, k, k] = temp_F_k_k[..., 0]
            # Number of flops below: (2*k - 1) * D_mybest.shape[1] * D_mybest.shape[2]
            # t1 = time.time()
            # D_mybest[k] = (D_mybest[:k].transpose([1, 2, 0]) @ D_mybest_maxindices[:, :, None]).squeeze(-1)  # <- faster than np.einsum('ibj,ib->bj', D_mybest[:k], D_mybest_maxindices)
            # t2 = time.time()
            # print(((2*k - 1) * D_mybest.shape[1] * D_mybest.shape[2] * 1e-9)/(t2-t1), 'GFLOPS')
            update_D_mybest_fast(temp_F_k_k[:, 0], XTX, maxindices, D_mybest[:k].transpose([1, 2, 0]), D_mybest_maxindices, D_mybest[k])
        a_F[k] = temp_F_k_k * np.take_along_axis(projections, maxindices[:, None], 1)
        update_projections_blast(projections, D_mybest[k], -a_F[k, :, 0])  # Around a 3x speedup :D
        # projections2 = projections + (-a_F[k]) * D_mybest[k]  # Relativeely slow as all the subsets are different...
        normr2 = normr2 - (a_F[k] * a_F[k]).squeeze(-1)
    else:
        np.put_along_axis(xests, gamma.T, (F @ a_F.squeeze(-1).T[:, :, None]).squeeze(-1), -1)
        # Instead of putting a stopping criteria, we could return the order in which the coeffs were added. (i.e. just return gamma and amplitudes)
    return xests

def omp_v0_new_blas(y, X, XTX, XTy, n_nonzero_coefs=None):
    if n_nonzero_coefs is None:
        n_nonzero_coefs = X.shape[1]

    B = y.shape[1]
    innerp = lambda x: np.einsum('ij,ij->i', x, x)
    normr2 = innerp(y.T)  # Norm squared of residual.
    projections = XTy

    gamma = np.zeros(shape=(n_nonzero_coefs, B), dtype=np.int64)
    F = np.repeat(np.identity(n_nonzero_coefs, dtype=X.dtype)[np.newaxis], B, 0)
    a_F = np.zeros_like(X, shape=(n_nonzero_coefs, B, 1))
    D_mybest = np.empty_like(X, shape=(n_nonzero_coefs, B, XTX.shape[1]))
    temp_F_k_k = 1
    xests = np.zeros((B, X.shape[1]))
    for k in range(n_nonzero_coefs):
        maxindices = get_max_projections_blas(projections)  # Numba version is about twice as fast as: np.argmax(projections * projections, 1)  # Maybe replace square with abs?
        gamma[k] = maxindices
        if k == 0:
            D_mybest[k] = XTX[maxindices]
        else:
            D_mybest_maxindices = np.take_along_axis(D_mybest[:k, :, :].transpose([2, 1, 0]), maxindices[None, :, None], 0).squeeze(0)
            temp_F_k_k = np.sqrt(1/(1-innerp(D_mybest_maxindices)))[:, None]
            update_D_mybest_blast(temp_F_k_k[:, 0], XTX, maxindices, D_mybest[:k].transpose([1, 2, 0]), D_mybest_maxindices, D_mybest[k])
            F[:, :, k] = -temp_F_k_k * (F[:, :, :k] @ D_mybest_maxindices[:, :, None]).squeeze(-1)  #
            F[:, k, k] = temp_F_k_k[..., 0]
            # t1 = time.time()
            # t2 = time.time()
            # print(D_mybest.shape[1] * ((2*k - 1) * D_mybest.shape[2] + 2 * D_mybest.shape[2])/(t2-t1) * 1e-9, 'GFLOPS')

        a_F[k] = temp_F_k_k * np.take_along_axis(projections, maxindices[:, None], 1)

        update_projections_blast(projections, D_mybest[k], -a_F[k, :, 0])  # Around a 3x speedup :D
        # projections2 = projections + (-a_F[k]) * D_mybest[k]  # Relativeely slow as all the subsets are different...
        normr2 = normr2 - (a_F[k] * a_F[k]).squeeze(-1)
    else:
        np.put_along_axis(xests, gamma.T, (F @ a_F.squeeze(-1).T[:, :, None]).squeeze(-1), -1)
        # Instead of putting a stopping criteria, we could return the order in which the coeffs were added. (i.e. just return gamma and amplitudes)
    return xests


def omp_v0_torch(y, X, n_nonzero_coefs=None):
    # FIXME: If it is cpu tensor, just call numpy version? (Keep two separate versions?)
    if n_nonzero_coefs is None:
        n_nonzero_coefs = X.shape[1]
    B = y.shape[1]
    innerp = lambda x: (x[:, None, :] @ x[:, :, None])[:, 0, 0]  # torch.einsum('ij,ij->i', x, x)
    normr2 = innerp(y.transpose(0, 1))  # Norm squared of residual.
    projections = (X.transpose(1, 0) @ y.transpose(1, 0)[:, :, None]).squeeze(-1)
    XTX = X.transpose(1, 0) @ X

    gamma = y.new_zeros(n_nonzero_coefs, B, dtype=torch.int64)
    F = torch.eye(n_nonzero_coefs, dtype=y.dtype, device=y.device).repeat(B, 1, 1)
    a_F = y.new_zeros(n_nonzero_coefs, B, 1)
    D_mybest = y.new_empty(B, XTX.shape[0], n_nonzero_coefs)
    xests = y.new_zeros(B, X.shape[1])
    temp_F_k_k = 1
    for k in range(n_nonzero_coefs):
        maxindices = projections.abs().argmax(1)
        gamma[k] = maxindices
        if k == 0:
            new_D_mybest = XTX[maxindices]
        else:
            grabdices = XTX.gather(0, maxindices[:, None].expand(-1, XTX.shape[1]))
            indices = maxindices + D_mybest.shape[1] * torch.arange(B, dtype=maxindices.dtype, device=maxindices.device)
            D_mybest_maxindices = D_mybest[:, :, :k].view(-1, k).gather(0, indices[:, None].expand(-1, k))
            # ^ LINEAR INDEXING!
            temp_F_k_k = torch.rsqrt(1 - innerp(D_mybest_maxindices))[:, None]  # Takes about 8% of time
            D_mybest_maxindices = -temp_F_k_k * D_mybest_maxindices  # minimal operations, exploit linearity
            new_D_mybest = temp_F_k_k * grabdices + (D_mybest[:, :, :k] @ D_mybest_maxindices[:, :, None]).squeeze(-1)
            F[:, :, k] = (F[:, :, :k] @ D_mybest_maxindices[:, :, None]).squeeze(-1)
            F[:, k, k] = temp_F_k_k[..., 0]

        a_F[k] = temp_F_k_k * torch.gather(projections, 1, maxindices[:, None])
        projections = projections - a_F[k] * new_D_mybest
        D_mybest[:, :, k] = new_D_mybest
        normr2 = normr2 - (a_F[k] * a_F[k]).squeeze(-1)
    else:
        values = (F @ a_F.squeeze(-1).transpose(1, 0)[:, :, None]).squeeze(-1)
        for i in range(B):
            xests[i].index_copy_(0, gamma[:, i], values[i])
    return xests

def update_D_mybest_fast(temp_F_k_k, XTX, maxindices, A, x, D_mybest):
    # D_mybest[...] = -temp_F_k_k * (A @ x[:, :, None]).squeeze(-1)
    # ^ This is a parallelized version of the first line in the loop below.
    for i in range(temp_F_k_k.shape[0]):
         D_mybest[i] = -temp_F_k_k[i] * (A[i] @ x[i, :, None]).squeeze(-1)  # dgemv
         D_mybest[i] = temp_F_k_k[i] * XTX[maxindices[i]] + D_mybest[i]  # daxpy

def batch_mm(matrix, matrix_batch):
    """
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    # From https://github.com/pytorch/pytorch/issues/14489#issuecomment-607730242
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose([1, 0, 2]).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return (matrix @ vectors).reshape(matrix.shape[0], batch_size, -1).transpose([1, 0, 2])

def faster_projections(Xt, r):
    # testB = Xt @ r[:, :, None]  Takes three times as long as the below function.
    return batch_mm(Xt, r[:, :, None])


def solve_lstsq(current_problems, A, y):
    # dgels (QR/LQ), dgelsy (QR) or dgelsd (SVD)
    solutions = np.empty_like(y, shape=[y.shape[0], current_problems.shape[-1]])
    # Cholesky:
    # TODO: Use dsyrk to calculate matrix times own transpose. Then solve with posv/gelsy.
    current_problemst = current_problems.transpose([0, 2, 1])
    b = (current_problemst @ y[:, :, None]).squeeze(-1)
    # We may want to use posv, which applies cholesky factorization.
    func = scipy.linalg.blas.find_best_blas_type(dtype=y.dtype)[0] + 'posv'  # Same naming convention as BLAS
    func = getattr(scipy.linalg.lapack, func)
    for i in range(y.shape[0]):
        solutions[i] = func(A[i], b[i], lower=True)[1]  # scipy.linalg.solve(A[i], b[i], sym_pos=True, assume_a="pos", overwrite_a=True, overwrite_b=True)
    return solutions

def omp_naive(X, y, n_nonzero_coefs):
    Xt = np.ascontiguousarray(X.T)
    y = np.ascontiguousarray(y.T)
    r = y.copy()  # Maybe no transpose? Remove this line?
    sets = np.zeros((n_nonzero_coefs, r.shape[0]), dtype=np.int32)
    problems = np.zeros((r.shape[0], n_nonzero_coefs, X.shape[0]))
    As = np.repeat(np.identity(n_nonzero_coefs, dtype=X.dtype)[np.newaxis], r.shape[0], 0)
    # solutions = np.zeros((r.shape[0], n_nonzero_coefs))
    xests = np.zeros_like(y, shape=(r.shape[0], X.shape[1]))
    for k in range(n_nonzero_coefs):
        #t1 = time.time()
        projections = faster_projections(Xt, r).squeeze(-1)  # X.shape[0] * (2*X.shape[1]-1) * r.shape[0] = O(bNM), where X is an MxN matrix, N>M.
        #t2 = time.time()
        #print((X.shape[0] * (2*X.shape[1]-1) * r.shape[0] * 1e-9)/(t2-t1), 'GFLOPS')
        best_idxs = get_max_projections_blas(projections)  # best_idxs = np.abs(projections).squeeze(-1).argmax(1)  # O(bN), https://scikit-cuda.readthedocs.io/en/latest/generated/skcuda.misc.maxabs.html
        sets[k, :] = best_idxs
        best = Xt[best_idxs, :]  # A mess...
        problems[:, k, :] = best
        current_problemst = problems[:, :k+1, :]
        current_problems = current_problemst.transpose([0, 2, 1])
        # TODO: We have already computed the result of current_problemst @ y[:, :, None]. (It is in projections I believe)
        #       And similarly for the hermitian - it can be constructed from XTX.
        # LAPACK has dgesvx/dgelsy (seems gelsy is newer/better) - gesv is for solving, these other ones work for least squares!
        #  I think linalg.solve calls gesv, which uses LU factorization. https://stackoverflow.com/questions/55367024/fastest-way-of-solving-linear-least-squares
        update = (current_problemst[:, :k, :] @ best[..., None]).squeeze(-1)
        # Our algorithm could be so much faster if the lhs were not (likely) all different.
        #  like in batch_mm, we could then treat the rhs as a single matrix.
        As[:, k, :k] = update  # We only have to update the lower triangle for sposv to work!
        if True:
            As[:, :k, k] = update
            solutions = np.linalg.solve(
                 As[:, :k+1, :k+1],  # As[:, :k+1, :k+1],
                 current_problemst @ y[:, :, None]).squeeze(-1)  # O(bk^2) memory.
        else:
            # ^ This is faster for small matrices (Python overhead most likely, but may also just be complexity)
            solutions = solve_lstsq(current_problems, As[:, :k+1, :k+1], y)
        r = y - (current_problems @ solutions[:, :, None]).squeeze(-1)

        # maybe memoize in case y is large, such that probability of repeats is significant.
        # If we can move all the batched stuff to RHS (using dense representations) it may be faster.
        # maybe memoize in case y is large, such that probability of repeats is significant. <- In case sparsity is low (e.g. < 32), this may be the fastest method, only limited by memory. (binomial(n, n*0.2))   else:
        # https://en.wikipedia.org/wiki/Singular_value_decomposition#Relation_to_eigenvalue_decomposition
        # https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        # Test if _^2 is faster than abs(_)
    else:
        # np.put_along_axis(xests.T, sets.T, solutions, 1)
        np.put_along_axis(xests, sets.T, solutions, 1)

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

    print("Settings used for the test: ")
    print("Number of Samples: " + str(n_samples))
    print("Number of Components: " + str(n_components))
    print("Number of Features: " + str(n_features))
    print("Number of Nonzero Coefficients: " + str(n_nonzero_coefs))
    print("\n")
    get_max_projections((X.T @ y.T[:, :, None]).squeeze(-1))

    print('Single core. New implementation of algorithm v0 (pytorch)')
    with elapsed_timer() as elapsed:
        xests_v0_new_torch = omp_v0_torch(torch.as_tensor(y.copy()), torch.as_tensor(X.copy()), n_nonzero_coefs)
    print('Samples per second:', n_samples/elapsed())
    print("\n")

    print('Single core. New implementation of algorithm v0. (blas)')
    with elapsed_timer() as elapsed:
        xests_v0_blas = omp_v0_new_blas(y.copy(), X.copy(), X.T @ X, (X.T @ y.T[:, :, None]).squeeze(-1), n_nonzero_coefs)
    print('Samples per second:', n_samples/elapsed())
    print("\n")

    print('error in new code (blas)', np.max(np.abs(xests_v0_blas - xests_v0_new_torch.numpy())))

    exit()

    print('Single core. Naive implementation.')
    with elapsed_timer() as elapsed:
        xests_naive = omp_naive(X.copy(), y.copy(), n_nonzero_coefs)
    print('Samples per second:', n_samples/elapsed())
    print("\n")


    print('Single core. Implementation of algorithm v0.')
    with elapsed_timer() as elapsed:
        xests_v0 = omp_v0(y.copy(), X.copy(), X.T @ X, (X.T @ y.T[:, :, None]).squeeze(-1), n_nonzero_coefs)
    print('Samples per second:', n_samples/elapsed())
    print("\n")



    exit()

    # precompute=True seems slower for single core. Dunno why.
    omp_args = dict(n_nonzero_coefs=n_nonzero_coefs, precompute=False, fit_intercept=False)

    # Single core
    print('Single core. Sklearn')
    omp = OrthogonalMatchingPursuit(**omp_args)
    with elapsed_timer() as elapsed:
        omp.fit(X, y)
    print('Samples per second:', n_samples/elapsed())
    print("\n")
    print(np.max(np.abs(xests_v0_new - omp.coef_)))

    err_naive = np.linalg.norm(y.T - (X @ xests_naive[:, :, None]).squeeze(-1), 2, 1)
    err_v0 = np.linalg.norm(y.T - (X @ xests_v0[:, :, None]).squeeze(-1), 2, 1)
    err_v0_new = np.linalg.norm(y.T - (X @ xests_v0_new[:, :, None]).squeeze(-1), 2, 1)
    err_sklearn = np.linalg.norm(y.T - (X @ omp.coef_[:, :, None]).squeeze(-1), 2, 1)
    avg_ylen = np.linalg.norm(y, 2, 0)
    # print(np.median(naive_err) / avg_ylen, np.median(scipy_err) / avg_ylen)
    plt.plot(np.sort(err_naive / avg_ylen), label='Naive')
    plt.plot(np.sort(err_v0 / avg_ylen), '.', label='v0')
    plt.plot(np.sort(err_v0_new / avg_ylen), '.', label='v0_new')
    plt.plot(np.sort(err_sklearn / avg_ylen), '--', label='SKLearn')
    plt.legend()
    plt.title("Distribution of relative errors.")
    plt.show()
    exit(0)
    # Multi core
    no_workers = 2 # os.cpu_count()
    # TODO: Gramian can be calculated once locally, and sent to each thread.
    print('Multi core. With', no_workers, "workers on", os.cpu_count(), "(logical) cores.")
    inputs = np.array_split(y, no_workers, axis=-1)
    with multiprocessing.Pool(no_workers, initializer=init_threads, initargs=(solveomp, X, omp_args)) as p:  # num_workers=0
        with elapsed_timer() as elapsed:
            result = p.map(solveomp, inputs)
    print('Samples per second:', n_samples / elapsed())

    # dataset = RandomSparseDataset(n_samples, n_components, n_features, n_nonzero_coefs)
    # sampler = torch.utils.data.sampler.BatchSampler(SequentialSampler(dataset), batch_size=2, drop_last=False)

    # y_est = (X @ omp.coef_[:, :, np.newaxis]).squeeze(-1)
    # residuals = np.linalg.norm(y.T - y_est, 2, 0)

    # plt.hist(residuals)
    # plt.show()

