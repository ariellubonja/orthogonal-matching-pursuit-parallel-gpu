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

# siz = 1024
# n_components, n_features = siz*8, siz
# n_nonzero_coefs = siz//4
# n_samples = 10

# TODO: Create branches - always assume pytorch, and use numpy where convenient/faster, otherwise both methods can give r (r^2) and projections - so branch on algorithm to use.

n_components, n_features = 1024, 100
n_nonzero_coefs = 17
n_samples = 8000 * 2

# FIXME: We should have the inputs be X, y, and optionally XTX, XTy etc (so the user can supply it pre-computed)
# For the naive torch version, it seems there are many options - e.g. give precomputed XTX and/or XTy,
# or calculate it runningly. (trading BK^2+BK memory for less compute)

# FIXME: We need to implement early stopping, and non-normalized A: https://www.sciencedirect.com/topics/engineering/orthogonal-matching-pursuit
#         - Non-normalized A means we get correlation with residual

# TODO: Maybe do early stopping by removing from batch (e.g. only when 1/2 or some other criteria), and capturing stopping-iter.
#       Then reconstruction may still just happen outside of the main function
#       - this seems to be the optimal way as solutions change every iteration, and then recomputing all these solutions one time at the end will not add a lot more compute.


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
        if k < 3:
            print(projections[0])
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

def get_max_projections_blas(projections, out=None):
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
        if out is None:
            out = np.empty((projections.shape[0],), dtype=np.int64)
        # print(projections.shape, result2.shape)
        # argmax_blas(projections, result2)  # seems to be about 10 times faster than the above
        argmax_blast(projections, out)
        return out


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

#TODO: Maybe make an OMP version that is optimized for just a single sample?
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
    # D_mybest = np.empty_like(X, shape=(n_nonzero_coefs, B, XTX.shape[1])).transpose([1, 0, 2])
    # D_mybest = np.empty_like(X, shape=(B, n_nonzero_coefs, XTX.shape[1]))
    D_mybest = np.empty_like(X, shape=(B, XTX.shape[1], n_nonzero_coefs)).transpose([0, 2, 1])
    b_indices = np.arange(B, dtype=gamma.dtype)

    temp_F_k_k = 1
    xests = np.zeros((B, X.shape[1]))
    for k in range(n_nonzero_coefs):
        # For SOMP one has to do abs(sum(...))) manually I think (or at least sasum followed by isamax/argmax)
        maxindices = get_max_projections_blas(projections)  # Numba version is about twice as fast as: np.argmax(projections * projections, 1)  # Maybe replace square with abs?
        gamma[k] = maxindices
        if k == 0:
            D_mybest[:, k] = XTX[maxindices]
        else:
            D_mybest_maxindices = D_mybest.transpose([0, 2, 1])[b_indices, gamma[k], :k]
            # updateA = XT[sets[k, :], :]

            temp_F_k_k = np.sqrt(1/(1-innerp(D_mybest_maxindices)))[:, None]
            D_mybest[b_indices, k] = XTX[maxindices]
            # .transpose([1, 2, 0])
            # print(D_mybest[:, k].shape, D_mybest[:, :k].transpose([0, 2, 1]).shape)
            update_D_mybest_blast(temp_F_k_k[:, 0], XTX, maxindices, D_mybest[:, :k].transpose([0, 2, 1]), D_mybest_maxindices, D_mybest[:, k])
            F[:, :, k] = -temp_F_k_k * (F[:, :, :k] @ D_mybest_maxindices[:, :, None]).squeeze(-1)  #
            F[:, k, k] = temp_F_k_k[..., 0]
        a_F[k] = temp_F_k_k * np.take_along_axis(projections, maxindices[:, None], 1)
        update_projections_blast(projections, D_mybest[:, k], -a_F[k, :, 0])  # Around a 3x speedup :D
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
    innerp = lambda x: (x[..., None, :] @ x[..., :, None])[:, 0, 0]  # torch.einsum('ij,ij->i', x, x)
    normr2 = innerp(y.transpose(0, 1))  # Norm squared of residual.
    projections = (X.transpose(1, 0) @ y.transpose(1, 0)[:, :, None]).squeeze(-1)
    XTX = X.transpose(1, 0) @ X

    gamma = y.new_zeros(n_nonzero_coefs, B, dtype=torch.int64)
    F = torch.eye(n_nonzero_coefs, dtype=y.dtype, device=y.device).repeat(B, 1, 1)
    a_F = y.new_zeros(n_nonzero_coefs, B, 1)
    D_mybest = y.new_empty(B, n_nonzero_coefs, XTX.shape[0])
    xests = y.new_zeros(B, X.shape[1])
    b_indices = torch.arange(B, dtype=gamma.dtype, device=gamma.device)
    temp_F_k_k = y.new_ones((B, 1))
    for k in range(n_nonzero_coefs):
        gamma[k] = projections.abs().argmax(1)
        torch.gather(XTX, 0, gamma[k, :, None].expand(-1, XTX.shape[1]), out=D_mybest[:, k, :])
        if k:
            D_mybest_maxindices = D_mybest.permute(0, 2, 1)[b_indices, gamma[k], :k]
            # print(D_mybest_maxindices)
            torch.rsqrt(1 - innerp(D_mybest_maxindices), out=temp_F_k_k[:, 0])
            # print(torch.any(torch.isnan(temp_F_k_k)))
            # It may be faster to also save or use 1/* and not just 1/sqrt(*) - since many places this is multiplied twice!
            D_mybest_maxindices *= -temp_F_k_k  # minimal operations, exploit linearity
            # print(D_mybest_maxindices.shape)
            D_mybest[:, k, :] *= temp_F_k_k
            D_mybest[:, k, :, None].baddbmm_(D_mybest[:, :k, :].permute(0, 2, 1), D_mybest_maxindices[:, :, None])
            torch.bmm(F[:, :, :k], D_mybest_maxindices[:, :, None], out=F[:, :, k, None])
            F[:, k, k] = temp_F_k_k[..., 0]

        a_F[k] = temp_F_k_k * torch.gather(projections, 1, gamma[k, :, None])
        normr2 -= (a_F[k] * a_F[k]).squeeze(-1)
        projections -= a_F[k] * D_mybest[:, k, :]
    else:
        # And here is the only place where F is used.
        # We could possibly do away with F and reconstruct x just from X (least squares with subset of X).
        values = (F @ a_F.squeeze(-1).transpose(1, 0)[:, :, None]).squeeze(-1)
        xests[b_indices[:, None], gamma.permute(1, 0)] = values
    return xests

def omp_v0_torch_2(y, X, n_nonzero_coefs=None):
    # FIXME: If it is cpu tensor, just call numpy version? (Keep two separate versions?)
    if n_nonzero_coefs is None:
        n_nonzero_coefs = X.shape[1]
    B = y.shape[1]
    innerp = lambda x: (x[..., None, :] @ x[..., :, None])[:, 0, 0]  # torch.einsum('ij,ij->i', x, x)
    normr2 = innerp(y.transpose(0, 1))  # Norm squared of residual.
    projections = (X.transpose(1, 0) @ y.transpose(1, 0)[:, :, None]).squeeze(-1)
    XTX = X.transpose(1, 0) @ X

    gamma = y.new_zeros(n_nonzero_coefs, B, dtype=torch.int64)
    F = torch.eye(n_nonzero_coefs, dtype=y.dtype, device=y.device).repeat(B, 1, 1)
    F2 = torch.eye(n_nonzero_coefs, dtype=y.dtype, device=y.device).repeat(B, 1, 1)
    a_F = y.new_zeros(n_nonzero_coefs, B, 1)
    D_mybest = y.new_empty(B, n_nonzero_coefs, XTX.shape[0])
    xests = y.new_zeros(B, X.shape[1])
    b_indices = torch.arange(B, dtype=gamma.dtype, device=gamma.device)
    temp_F_k_k = y.new_ones((B, 1))
    for k in range(n_nonzero_coefs):
        gamma[k] = projections.abs().argmax(1)
        torch.gather(XTX, 0, gamma[k, :, None].expand(-1, XTX.shape[1]), out=D_mybest[:, k, :])
        if k:
            D_mybest_maxindices = D_mybest.permute(0, 2, 1)[b_indices, gamma[k], :k]
            # print(D_mybest_maxindices)
            torch.rsqrt(1 - innerp(D_mybest_maxindices), out=temp_F_k_k[:, 0])
            # print(torch.any(torch.isnan(temp_F_k_k)))
            # It may be faster to also save or use 1/* and not just 1/sqrt(*) - since many places this is multiplied twice!
            D_mybest_maxindices *= -temp_F_k_k  # minimal operations, exploit linearity
            # print(D_mybest_maxindices.shape)
            D_mybest[:, k, :] *= temp_F_k_k
            D_mybest[:, k, :, None].baddbmm_(D_mybest[:, :k, :].permute(0, 2, 1), D_mybest_maxindices[:, :, None])
            F[:, k, k] = temp_F_k_k[..., 0]
            F2[:, k, k] = temp_F_k_k[..., 0]
            F2[:, :k, k] = D_mybest_maxindices
            torch.bmm(F[:, :k, :k], D_mybest_maxindices[:, :, None], out=F[:, :k, k, None])
            print((F[:, :k, :k] @ D_mybest_maxindices[:, :, None])[0])
            print(F2[0])

        a_F[k] = temp_F_k_k * torch.gather(projections, 1, gamma[k, :, None])
        normr2 -= (a_F[k] * a_F[k]).squeeze(-1)
        projections -= a_F[k] * D_mybest[:, k, :]
    else:
        # And here is the only place where F is used.
        # We could possibly do away with F and reconstruct x just from X (least squares with subset of X).
        values = (F @ a_F.squeeze(-1).transpose(1, 0)[:, :, None]).squeeze(-1)
        xests[b_indices[:, None], gamma.permute(1, 0)] = values
    return xests

def update_D_mybest_fast(temp_F_k_k, XTX, maxindices, A, x, D_mybest):
    # D_mybest[...] = -temp_F_k_k * (A @ x[:, :, None]).squeeze(-1)
    # ^ This is a parallelized version of the first line in the loop below.
    for i in range(temp_F_k_k.shape[0]):
         D_mybest[i] = -temp_F_k_k[i] * (A[i] @ x[i, :, None]).squeeze(-1)  # dgemv
         D_mybest[i] = temp_F_k_k[i] * XTX[maxindices[i]] + D_mybest[i]  # daxpy


def faster_projections(Xt, r):
    # testB = Xt @ r[:, :, None]  Takes three times as long as the below function.
    return batch_mm(Xt, r[:, :, None])


def omp_naive_torch(X, y, n_nonzero_coefs):
    y = y.transpose(1, 0).contiguous()
    XT = X.transpose(1, 0)
    SOMP = len(y.shape) == 3
    if not SOMP:
        y = y.unsqueeze(-1)

    r = y.clone()  # Maybe no transpose? Remove this line?
    sets = r.new_zeros(n_nonzero_coefs, r.shape[0], dtype=torch.int64)
    As = r.new_zeros(r.shape[0], n_nonzero_coefs, X.shape[0])
    # Trade b*k^2 memory for much less compute time. (This has to be done anyways)
    ATAs = torch.eye(n_nonzero_coefs, dtype=X.dtype, device=X.device).repeat(r.shape[0], 1, 1)
    # Trade b*k memory for less compute time. (This has to be done anyways)
    ATys = r.new_zeros(y.shape[0], n_nonzero_coefs, y.shape[2])
    xests = r.new_zeros(r.shape[0], X.shape[1])
    for k in range(n_nonzero_coefs):
        sets[k, :] = (XT @ r).abs().sum(-1).argmax(-1)  # Store in sets.
        # TODO: CALL IDAMAX ON GPU? or do like this: https://scikit-cuda.readthedocs.io/en/latest/generated/skcuda.misc.maxabs.html
        #       Alternatively do like scikit-cuda, but add the sum step to the kernel as well.
        # We will now update the following:
        AT = As[:, :k + 1, :]
        A = AT.permute(0, 2, 1)
        ATA = ATAs[:, :k + 1, :k + 1]
        ATy = ATys[:, :k + 1]

        # Update As with the new column we add.
        updateA = XT[sets[k, :], :]
        As[:, k, :] = updateA
        # Update ATy
        ATys[:, k, :] = (updateA[:, None] @ y).squeeze(-2)
        # Update ATA
        ATAs[:, :k, k] = torch.bmm(AT[:, :k, :], updateA[:, :, None]).squeeze(-1)
        ATAs[:, k, :k] = ATAs[:, :k, k]

        # Done updating, now solve the pseudo-inverse by the exact solution to ATAx=ATy.
        if False:
            solutions = torch.solve(ATy, ATA)[0]
        else:
            # We only have to store one triangle of ATA for this to work.
            factors = torch.linalg.cholesky(ATA)
            solutions = torch.cholesky_solve(ATy, factors)
        # ^ We also tried using cholesky solve - it gives around 1% speed-up overall.
        torch.baddbmm(y, A, solutions, beta=-1, out=r)
    else:
        if SOMP:
            return sets.permute(1, 0)
        else:
            xests[torch.arange(r.shape[0], dtype=sets.dtype, device=sets.device)[:, None],
                  sets.permute(1, 0)] = solutions.squeeze(-1)
    return xests

def omp_naive_torch(X, y, n_nonzero_coefs):
    XT = X.transpose(1, 0)
    y = y.transpose(1, 0).contiguous()
    if len(y.shape) == 2:
        y = y.unsqueeze(-1)
    r = y.clone()  # Maybe no transpose? Remove this line?
    sets = r.new_zeros(n_nonzero_coefs, r.shape[0], dtype=torch.int64)
    As = r.new_zeros(r.shape[0], n_nonzero_coefs, X.shape[0])
    # Trade b*k^2 memory for much less compute time. (This is the minimum memory requirement for dense)
    ATAs = torch.eye(n_nonzero_coefs, dtype=X.dtype, device=X.device).repeat(r.shape[0], 1, 1)
    # Trade b*k memory for less compute time.
    ATys = r.new_zeros(y.shape[0], n_nonzero_coefs, y.shape[2])
    xests = r.new_zeros(r.shape[0], X.shape[1])
    for k in range(n_nonzero_coefs):
        sets[k, :] = torch.as_tensor(get_max_projections_blas((XT @ r).squeeze(-1).numpy())) # https://scikit-cuda.readthedocs.io/en/latest/generated/skcuda.misc.maxabs.html
        # sets[k, :] = (XT @ r).abs().sum(-1).argmax(-1)  # Store in sets.
        # Update As with the new column we add.
        updateA = XT[sets[k, :], :]
        As[:, k, :] = updateA
        AT = As[:, :k + 1, :]
        A = AT.permute(0, 2, 1)
        # Update ATy
        ATys[:, k, :] = (updateA[:, None] @ y).squeeze(-2)
        ATy = ATys[:, :k + 1]
        # Update ATA
        ATAs[:, :k, k] = torch.bmm(AT[:, :k, :], updateA[:, :, None]).squeeze(-1)
        ATAs[:, k, :k] = ATAs[:, :k, k]
        ATA = ATAs[:, :k + 1, :k + 1]

        solutions = torch.solve(ATy, ATA)[0]
        # ^ We also tried using cholesky solve - it gives around 1% speed-up overall.
        torch.baddbmm(y, A, solutions, beta=-1, out=r)
    else:
        xests[torch.arange(r.shape[0], dtype=sets.dtype, device=sets.device)[:, None],
              sets.permute(1, 0)] = solutions.squeeze(-1)
    return xests


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


def batch_mm(matrix, matrix_batch, return_contiguous=True):
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
    if return_contiguous:
        result = np.empty_like(matrix_batch, shape=(batch_size, matrix.shape[0], matrix_batch.shape[2]))
        np.matmul(matrix, vectors, out=result.transpose([1, 0, 2]).reshape(matrix.shape[0], -1))
    else:
        result = (matrix @ vectors).reshape(matrix.shape[0], batch_size, -1).transpose([1, 0, 2])

    return result

def omp_naive(X, y, n_nonzero_coefs):
    XT = np.asfortranarray(X.T)
    y = np.ascontiguousarray(y.T)  # TODO: Maybe this is not the fastest way to do it
    r = y.copy()
    sets = np.zeros((n_nonzero_coefs, y.shape[0]), dtype=np.int32)

    ATs = np.empty_like(y, shape=(y.shape[0], n_nonzero_coefs, X.shape[0]))
    # Trade b*k^2+bk memory for much less compute time. (This has to be done anyways, since we are batching, otherwise one could just permute columns of X in-place, as in https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/linear_model/_omp.py#L28 )
    ATAs = np.empty_like(r, shape=(y.shape[0], n_nonzero_coefs, n_nonzero_coefs))
    ATAs[:] = np.identity(n_nonzero_coefs, dtype=ATAs.dtype)

    # ATAsPacked = ATAs[(b_indices,) + np.triu_indices(n_nonzero_coefs)].T.copy()
    tri_idx = np.tril_indices(n_nonzero_coefs)
    ATAsPacked = ATAs[:, tri_idx[0], tri_idx[1]].transpose([1, 0])
    ATys = np.zeros_like(r, shape=(y.shape[0], n_nonzero_coefs, 1))  # .transpose([1, 0, 2])
    xests = np.zeros_like(y, shape=(y.shape[0], X.shape[1]))
    innerp = lambda x, y=None, out=None: np.einsum('bi,bi->b', x, x if y is None else y, out=out)
    for k in range(n_nonzero_coefs):
        projections = batch_mm(XT, r[:, :, None]).squeeze(-1)
        sets[k, :] = get_max_projections_blas(projections)
        # We will now update the following:
        AT = ATs[:, :k + 1, :] # A.transpose([0, 2, 1])
        ATA = ATAs[:, :k + 1, :k + 1]
        ATy = ATys[:, :k + 1]

        # Update As with the new column we add.
        updateA = XT[sets[k, :], :]
        AT[:, k, :] = updateA
        # Update ATy
        innerp(updateA, y, out=ATy[:, k, 0])

        if True:
            # Update ATAsPacked
            packed_idx = k*(k-1)//2
            np.matmul(AT[:, :k+1, :], updateA[:, :, None], out=ATAsPacked[k + packed_idx:packed_idx+2*k+1, :].T[:, :, None])
            solutions = ATy.transpose([0, 2, 1]).copy().transpose([0, 2, 1])  # We need it in fortran order.
            ppsv(np.ascontiguousarray(ATAsPacked[:packed_idx+2*k+1, :].T), solutions)
        else:
            # Update ATAs
            np.matmul(AT[:, :k+1, :], updateA[:, :, None], out=ATA[:, k, :k+1, None])  # We could use the following to dynamically select order: ATA[:, k, :k].strides[-1], ATA[:, :k, k].strides[-1]
            ATA[:, :k, k] = ATA[:, k, :k]
            solutions = np.linalg.solve(ATA, ATy)

        r[:] = y - (AT.transpose([0, 2, 1]) @ solutions).squeeze(-1)

        # maybe memoize in case y is large, such that probability of repeats is significant.
        # If we can move all the batched stuff to RHS (using dense representations) it may be faster.
        # maybe memoize in case y is large, such that probability of repeats is significant. <- In case sparsity is low (e.g. < 32), this may be the fastest method, only limited by memory. (binomial(n, n*0.2))   else:
        # https://en.wikipedia.org/wiki/Singular_value_decomposition#Relation_to_eigenvalue_decomposition
        # https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        # Test if _^2 is faster than abs(_)
    else:
        xests[np.arange(r.shape[0], dtype=sets.dtype)[:, None], sets.transpose([1, 0])] = solutions.squeeze(-1)

    return xests


def batch_mm_torch(matrix, matrix_batch, return_contiguous=True):
    """
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    # From https://github.com/pytorch/pytorch/issues/14489#issuecomment-607730242
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.permute(1, 0, 2).view(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    if return_contiguous:
        result = matrix_batch.new_empty(batch_size, matrix.shape[0], matrix_batch.shape[2])
        torch.matmul(matrix, vectors, out=result.permute(1, 0, 2).view(matrix.shape[0], -1))
    else:
        result = (matrix @ vectors).view(matrix.shape[0], batch_size, -1).permute(1, 0, 2)

    return result

def omp_naive_final(X, y, n_nonzero_coefs):
    XT = X.clone().t()  # Store in fortran-order.
    y = y.t().contiguous()  # TODO: Maybe this is not the fastest way to do it
    r = y.clone()
    sets = y.new_zeros((n_nonzero_coefs, y.shape[0]), dtype=torch.long)

    B = r.shape[0]
    M = X.shape[0]
    N = X.shape[1]

    # Trade b*k^2+bk+bkM = O(bkM) memory for much less compute time. (This has to be done anyways, since we are batching, otherwise one could just permute columns of X in-place, as in https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/linear_model/_omp.py#L28 )
    #                    (This means our algorithm also uses O(bkM+bMN) = O(bMN) memory)
    ATs = y.new_empty(B, n_nonzero_coefs, M)
    ATys = y.new_empty(B, n_nonzero_coefs, 1)  # .transpose([1, 0, 2])
    ATAs = y.new_empty(B, n_nonzero_coefs, n_nonzero_coefs)
    ATAs[:] = torch.eye(n_nonzero_coefs, dtype=ATAs.dtype, device=ATAs.device)

    tri_idx = torch.tril_indices(n_nonzero_coefs, n_nonzero_coefs)  # TODO: Must it be on the same device as ATA?
    ATAsPacked = ATAs[:, tri_idx[0], tri_idx[1]].t()

    xests = y.new_empty(B, N)
    innerp = lambda x, y, out=None: torch.matmul(x[..., None, :], y[..., :, None], out=None if out is None else out[:, None, None])

    for k in range(n_nonzero_coefs):
        projections = batch_mm_torch(XT, r[:, :, None]).squeeze(-1)
        if not projections.is_cuda:
            get_max_projections_blas(projections.numpy(), sets[k, :].numpy())
        else:
            # sum does nothing but squeeze, but would be relevant in SOMP.
            sets[k, :] = projections.abs().sum(-1).argmax(-1)

        # We will now update the following:
        ATA = ATAs[:, :k + 1, :k + 1]  # (or we'll be updating the packed version of it)

        # Update As with the new column we add.
        AT = ATs[:, :k + 1, :]
        updateA = XT[sets[k, :], :]
        AT[:, k, :] = updateA
        # Update ATy
        ATy = ATys[:, :k + 1]
        innerp(updateA, y, out=ATy[:, k, 0])

        if not ATAsPacked.is_cuda:
            # Update ATAsPacked
            packed_idx = k * (k - 1) // 2
            np.matmul(AT[:, :k + 1, :].numpy(), updateA[:, :, None].numpy(), out=ATAsPacked[k + packed_idx:packed_idx + 2 * k + 1, :].t()[:, :, None].numpy())
            # ^ For some reason numpy is much faster (on my PC at least)
            # torch.bmm(AT[:, :k + 1, :], updateA[:, :, None], out=ATAsPacked[k + packed_idx:packed_idx + 2 * k + 1, :].t()[:, :, None])

            # Time to solve the normal equations (using cholesky factorization with packed input)
            solutions = ATy.permute(0, 2, 1).clone().permute(0, 2, 1)  # We need it in fortran order.
            ppsv(ATAsPacked[:packed_idx + 2 * k + 1, :].t().contiguous().numpy(), solutions.numpy())
        else:
            # Update ATAs
            torch.bmm(AT[:, :k + 1, :], updateA[:, :, None], out=ATA[:, k, :k + 1, None])  # We could use the following to dynamically select order: ATA[:, k, :k].strides[-1], ATA[:, :k, k].strides[-1]
            if True:
                # Use cholesky decomposition to solve.
                factors = torch.cholesky(ATA)  # or torch.linalg.cholesky!
                solutions = torch.cholesky_solve(ATy, factors)
            else:
                # Time to solve the normal equations
                ATA[:, :k, k] = ATA[:, k, :k]  # We have to store the other triangle as well then.
                solutions = torch.solve(ATy, ATA)[0]

        if ATAsPacked.is_cuda:
            torch.baddbmm(y[:, :, None], AT.permute(0, 2, 1), solutions, beta=-1, out=r[:, :, None])
        else:
            # Numpy matmul seems faster.
            r.numpy()[:] = y.numpy() - (AT.permute(0, 2, 1).numpy() @ solutions.numpy()).squeeze(-1)

    else:
        xests[torch.arange(r.shape[0], dtype=sets.dtype, device=sets.device)[:, None], sets.t()] = solutions.squeeze(-1)

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


    print('Single core. Naive implementation.')
    with elapsed_timer() as elapsed:
        xests_naive = omp_naive(X.copy(), y.copy(), n_nonzero_coefs)
    print('Samples per second:', n_samples / elapsed())
    print("\n")

    print('Single core. Naive implementation.')
    with elapsed_timer() as elapsed:
        xests_naive = omp_naive(X.copy(), y.copy(), n_nonzero_coefs)
    print('Samples per second:', n_samples / elapsed())
    print("\n")

    if True:
        print('Single core. Naive implementation.')
        with elapsed_timer() as elapsed:
            xests_naive_final = omp_naive_final(torch.as_tensor(X.copy(), dtype=torch.float64), torch.as_tensor(y.copy(), dtype=torch.float64), n_nonzero_coefs).cpu().numpy()
        print('Samples per second:', n_samples / elapsed())
        print("\n")

        print('Single core. Naive implementation.')
        with elapsed_timer() as elapsed:
            xests_naive_final = omp_naive_final(torch.as_tensor(X.copy(), dtype=torch.float64), torch.as_tensor(y.copy(), dtype=torch.float64), n_nonzero_coefs).cpu().numpy()
        print('Samples per second:', n_samples / elapsed())
        print("\n")

    print(xests_naive_final.shape, xests_naive.shape)
    print(np.max(np.abs(xests_naive_final-xests_naive)))

    exit(0)
    print('Single core. New implementation of algorithm v0. (blas)')
    with elapsed_timer() as elapsed:
        xests_v0_blas = omp_v0_new_blas(y.copy(), X.copy(), X.T @ X, (X.T @ y.T[:, :, None]).squeeze(-1), n_nonzero_coefs)
    print('Samples per second:', n_samples/elapsed())
    print("\n")


    print('Single core. New implementation of algorithm v0. (blas)')
    with elapsed_timer() as elapsed:
        xests_v0_blas = omp_v0_new_blas(y.copy(), X.copy(), X.T @ X, (X.T @ y.T[:, :, None]).squeeze(-1), n_nonzero_coefs)
    print('Samples per second:', n_samples/elapsed())
    print("\n")

    print(np.max(np.abs(xests_v0_blas-xests_naive)))

    exit(0)

    omp_args = dict(n_nonzero_coefs=n_nonzero_coefs, precompute=False, fit_intercept=False)
    # Single core
    print('Single core. Sklearn')
    omp = OrthogonalMatchingPursuit(**omp_args)
    with elapsed_timer() as elapsed:
        omp.fit(X, y)
    print('Samples per second:', n_samples / elapsed())
    print("\n")

    # Single core
    print('Single core. Sklearn (float32)')
    omp = OrthogonalMatchingPursuit(**omp_args)
    with elapsed_timer() as elapsed:
        omp.fit(X.astype(np.float32), y.astype(np.float32))
    print('Samples per second:', n_samples / elapsed())
    print("\n")

    print('Single core. New implementation of algorithm v0. (blas)')
    with elapsed_timer() as elapsed:
        xests_v0_blas = omp_v0_new_blas(y.copy(), X.copy(), X.T @ X, (X.T @ y.T[:, :, None]).squeeze(-1), n_nonzero_coefs)
    print('Samples per second:', n_samples/elapsed())
    print("\n")

    print('Single core. New implementation of algorithm v0 (pytorch)')
    with elapsed_timer() as elapsed:
        xests_v0_new_torch = omp_v0_torch_2(torch.as_tensor(y.copy(), dtype=torch.float32), torch.as_tensor(X.copy(), dtype=torch.float32), n_nonzero_coefs)
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

