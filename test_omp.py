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

n_components, n_features = 512*4, 100*4
n_nonzero_coefs = 17
n_samples = 2000

def solveomp(y):
    solveomp.omp.fit(solveomp.X, y)
    return solveomp.omp.coef_

def init_threads(func, X, omp_args):
    func.omp = OrthogonalMatchingPursuit(**omp_args)
    func.X = X

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

from numba import jit
@jit(nopython=True)
def get_max_projections(projections):
    # It seems BLAS ISAMAX/IDMAX should cover this? (Good for CUDA especially)
    B = projections.shape[0]
    result = np.zeros((B,), dtype=np.int64)
    for i in range(B):
        result[i] = np.argmax(np.abs(projections[i]))
    return result

import scipy
def get_max_projections_blas(projections):
    #  BLAS is even faster! :O
    func = 'i' + scipy.linalg.blas.find_best_blas_type(dtype=projections.dtype)[0] + 'amax'
    func = getattr(scipy.linalg.blas, func)
    B = projections.shape[0]
    result = np.zeros((B,), dtype=np.int64)
    for i in range(B):  # This has around a 20% Python overhead.
        result[i] = func(projections[i])  # np.argmax(np.abs(projections[i]))
    return result

def update_projections_blas(projections, D_mybest, coefs):
    func = scipy.linalg.blas.find_best_blas_type(dtype=projections.dtype)[0] + 'axpy'
    func = getattr(scipy.linalg.blas, func)
    result = np.zeros_like(projections)
    for i in range(projections.shape[0]):
        func(D_mybest[i], projections[i], a=coefs[i])  # np.argmax(np.abs(projections[i]))
    return result

@jit(nopython=True)
def update_D_mybest(temp_F_k_k, XTX, maxindices, einssum):
    for i in range(temp_F_k_k.shape[0]):
         einssum[i] = XTX[maxindices[i]] - einssum[i]
    einssum *= temp_F_k_k

def update_D_mybest_new(temp_F_k_k, XTX, maxindices, D_mybest, D_mybest_maxindices):
    # temp_F_k_k * (XTX[maxindices, :] - (D_mybest @ D_mybest_maxindices[:, :, None]).squeeze(-1))
    func = scipy.linalg.blas.find_best_blas_type(dtype=temp_F_k_k.dtype)[0] + 'gemv'
    func = getattr(scipy.linalg.blas, func)

    result = np.zeros_like(temp_F_k_k, shape=(temp_F_k_k.shape[0], XTX.shape[0]))  # Uses around 8% of time.
    for i in range(temp_F_k_k.shape[0]):  # 2% Python overhead it seems.
         result[i] = func(alpha=-temp_F_k_k[i], beta=temp_F_k_k[i], a=D_mybest[i], x=D_mybest_maxindices[i], y=XTX[maxindices[i]])
    return result

@profile
def omp_v0_new(y, X, XTX, XTy, n_nonzero_coefs=None):
    if n_nonzero_coefs is None:
        n_nonzero_coefs = X.shape[1]

    N = y.shape[1]
    innerp = lambda x: np.einsum('ij,ij->i', x, x)
    normr2 = innerp(y.T)  # Norm squared of residual.
    projections = XTy

    gamma = np.zeros(shape=(n_nonzero_coefs, N), dtype=np.int64)
    F = np.repeat(np.identity(n_nonzero_coefs, dtype=X.dtype)[np.newaxis], N, 0)
    a_F = np.zeros_like(X, shape=(n_nonzero_coefs, N, 1))
    D_mybest = np.zeros_like(X, shape=(n_nonzero_coefs, N, XTX.shape[0]))
    temp_F_k_k = 1
    xests = np.zeros((N, X.shape[1]))
    for k in range(n_nonzero_coefs):
        maxindices = get_max_projections_blas(projections)  # Numba version is about twice as fast as: np.argmax(projections * projections, 1)  # Maybe replace square with abs?
        # maxindices2 = np.argmax(projections * projections, 1)
        gamma[k] = maxindices
        if k == 0:
            D_mybest[k] = XTX[None, maxindices, :]
        else:
            # Do something about this:
            D_mybest_maxindices = np.take_along_axis(D_mybest[:k, :, :], maxindices[None, :, None], 2).squeeze(2)
            temp_F_k_k = np.sqrt(1/(1-innerp(D_mybest_maxindices.T)))[:, None]
            F[:, :, k] = -temp_F_k_k * (F[:, :, :k] @ D_mybest_maxindices.T[:, :, None]).squeeze(-1)
            F[:, k, k] = temp_F_k_k[..., 0]
            D_mybest[k] = np.einsum('ibj,ib->bj', D_mybest[:k], D_mybest_maxindices)
            update_D_mybest(temp_F_k_k, XTX, maxindices, D_mybest[k])
        a_F[k] = temp_F_k_k * np.take_along_axis(projections, maxindices[:, None], 1)
        update_projections_blas(projections, D_mybest[k], -a_F[k, :, 0])  # Relativeely slow as all the subsets are different...
        # projections2 = projections + (-a_F[k]) * D_mybest[k]  # Relativeely slow as all the subsets are different...
        normr2 = normr2 - (a_F[k] * a_F[k]).squeeze(-1)
    else:
        np.put_along_axis(xests, gamma.T, (F @ a_F.squeeze(-1).T[:, :, None]).squeeze(-1), -1)
        # Instead of putting a stopping criteria, we could return the order in which the coeffs were added. (i.e. just return gamma and amplitudes)
    # F[:, :k, :k] @ a_F.T[:k, :, None]
    return xests

@profile
def omp_v0_newer(y, X, XTX, XTy, n_nonzero_coefs=None):
    if n_nonzero_coefs is None:
        n_nonzero_coefs = X.shape[1]

    N = y.shape[1]
    innerp = lambda x: np.einsum('ij,ij->i', x, x)
    normr2 = innerp(y.T)  # Norm squared of residual.
    projections = XTy

    gamma = np.zeros(shape=(n_nonzero_coefs, N), dtype=np.int64)
    F = np.repeat(np.identity(n_nonzero_coefs, dtype=X.dtype)[np.newaxis], N, 0)
    a_F = np.zeros_like(X, shape=(n_nonzero_coefs, N, 1))
    D_mybest = np.zeros_like(X, shape=(N, XTX.shape[0], n_nonzero_coefs))
    temp_F_k_k = 1
    xests = np.zeros((N, X.shape[1]))
    for k in range(n_nonzero_coefs):
        maxindices = get_max_projections_blas(projections)  # Maybe replace square with abs?
        gamma[k] = maxindices
        if k == 0:
            new_D_mybest = XTX[maxindices, :]
        else:
            D_mybest_maxindices = np.take_along_axis(D_mybest[:, :, :k], maxindices[:, None, None], 1).squeeze(1)
            temp_F_k_k = np.sqrt(1/(1-innerp(D_mybest_maxindices)))[:, None]
            F[:, :, k] = -temp_F_k_k * (F[:, :, :k] @ D_mybest_maxindices[:, :, None]).squeeze(-1)
            F[:, k, k] = temp_F_k_k[..., 0]
            new_D_mybest = update_D_mybest_new(temp_F_k_k, XTX, maxindices, D_mybest[:, :, :k], D_mybest_maxindices)
        D_mybest[:, :, k] = new_D_mybest
        a_F[k] = temp_F_k_k * np.take_along_axis(projections, maxindices[:, None], 1)  # TODO: Figure out if we should use XTy.T or XTy, which is faster?
        update_projections_blas(projections, new_D_mybest, -a_F[k, :, 0])
        normr2 = normr2 - (a_F[k] * a_F[k]).squeeze(-1)
    else:
        np.put_along_axis(xests, gamma.T, (F @ a_F.squeeze(-1).T[:, :, None]).squeeze(-1), -1)
        # Instead of putting a stopping criteria, we could return the order in which the coeffs were added. (i.e. just return gamma and amplitudes)
    # F[:, :k, :k] @ a_F.T[:k, :, None]
    return xests

def omp_naive(X, y, n_nonzero_coefs):
    Xt = np.ascontiguousarray(X.T)
    y = np.ascontiguousarray(y.T)
    r = y.copy()  # Maybe no transpose? Remove this line?
    sets = np.zeros((n_nonzero_coefs, r.shape[0]), dtype=np.int32)
    problems = np.zeros((r.shape[0], X.shape[0], n_nonzero_coefs))
    solutions = np.zeros((r.shape[0], n_nonzero_coefs))
    xests = np.zeros((r.shape[0], X.shape[1]))
    for k in range(n_nonzero_coefs):
        projections = Xt @ r[:, :, None]  # O(bNM), where X is an MxN matrix, N>M.
        best_idxs = np.abs(projections).squeeze(-1).argmax(1)  # O(bN), https://scikit-cuda.readthedocs.io/en/latest/generated/skcuda.misc.maxabs.html
        sets[k, :] = best_idxs
        problems[:, :, k] = Xt[best_idxs, :]
        current_problems = problems[:, :, :k+1]
        current_problemst = current_problems.transpose([0, 2, 1])
        solutions = np.linalg.solve(current_problemst @ current_problems,  # O(bk^2) memory.
                                    current_problemst @ y[:, :, None]).squeeze(-1)
        r = y - (current_problems @ solutions[:, :, None]).squeeze(-1)
        # maybe memoize in case y is large, such that probability of repeats is significant.
        # If we can move all the batched stuff to RHS (using dense representations) it may be faster.
        # maybe memoize in case y is large, such that probability of repeats is significant. <- In case sparsity is low (e.g. < 32), this may be the fastest method, only limited by memory. (binomial(n, n*0.2))   else:
        # https://en.wikipedia.org/wiki/Singular_value_decomposition#Relation_to_eigenvalue_decomposition
        # https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        # Test if _^2 is faster than abs(_)
    else:
        np.put_along_axis(xests, sets.T, solutions, -1)
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
    # TODO: Warm up update_D_mybest(...) as well, for profiling
    print('Single core. New implementation of algorithm v0.')
    with elapsed_timer() as elapsed:
        xests_v0_new = omp_v0_newer(y, X, X.T @ X, (X.T @ y.T[:, :, None]).squeeze(-1), n_nonzero_coefs)
    print('Samples per second:', n_samples/elapsed())
    print("\n")

    print('Single core. Implementation of algorithm v0.')
    with elapsed_timer() as elapsed:
        xests_v0 = omp_v0_new(y, X, X.T @ X, (X.T @ y.T[:, :, None]).squeeze(-1), n_nonzero_coefs)
    print('Samples per second:', n_samples/elapsed())
    print("\n")

    print(np.max(np.abs(xests_v0_new - xests_v0)))
    exit(0)
    print('Single core. Naive Implementation, based on our Homework.')
    with elapsed_timer() as elapsed:
        xests_naive = omp_naive(X, y, n_nonzero_coefs)
    print('Samples per second:', n_samples/elapsed())
    print("\n")

    # precompute=True seems slower for single core. Dunno why.
    omp_args = dict(n_nonzero_coefs=n_nonzero_coefs, precompute=False, fit_intercept=False)

    # Single core
    print('Single core. Sklearn')
    omp = OrthogonalMatchingPursuit(**omp_args)
    with elapsed_timer() as elapsed:
        omp.fit(X, y)
    print('Samples per second:', n_samples/elapsed())
    print("\n")

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

