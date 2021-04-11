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
n_samples = 2500

def solveomp(y):
    solveomp.omp.fit(solveomp.X, y)
    return solveomp.omp.coef_

def init_threads(func, X, omp_args):
    func.omp = OrthogonalMatchingPursuit(**omp_args)
    func.X = X

# kernprof -l -v test_omp.py and @profile
# Based on https://github.com/zhuhufei/OMP/blob/master/codeAug2020.m
# From "Efficient Implementations for Orthogonal Matching Pursuit" (2020)
@profile
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
            D_mybest[:, :, k] = XTX[:, maxindices].T
        else:
            D_mybest_maxindices = np.take_along_axis(D_mybest, maxindices[:, None, None], 1).squeeze(1)
            temp_F_k_k = np.sqrt(1/(1-innerp(D_mybest_maxindices)))[:, None]
            F[:, :, k] = -temp_F_k_k * (F @ D_mybest_maxindices[:, :, None]).squeeze(-1)
            F[:, k, k] = temp_F_k_k[..., 0]

            D_mybest[:, :, k] = temp_F_k_k * (XTX[:, maxindices].T - (D_mybest @ D_mybest_maxindices[:, :, None]).squeeze(-1)) # To translate D_mybest*D_mybest(newgam,:).' from MATLAB. It seems to just be a matmul.
        a_F[k] = temp_F_k_k * np.take_along_axis(projections, maxindices[:, None], 1)  # TODO: Figure out if we should use XTy.T or XTy, which is faster?
        projections = projections - D_mybest[:, :, k] * a_F[k]
        normr2 = normr2 - (a_F[k] * a_F[k]).squeeze(-1)
    else:
        np.put_along_axis(xests, gamma.T, (F @ a_F.squeeze(-1).T[:, :, None]).squeeze(-1), -1)
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
        projections = Xt @ r[:, :, None]
        best_idxs = np.abs(projections).squeeze(-1).argmax(1)
        sets[k, :] = best_idxs
        problems[:, :, k] = Xt[best_idxs, :]
        current_problems = problems[:, :, :k+1]
        current_problemst = current_problems.transpose([0, 2, 1])
        # Memory complexity is k*N*M <= N^2 * M
        solutions = np.linalg.solve(current_problemst @ current_problems,
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

    print('Single core. Implementation of algorithm v0.')
    with elapsed_timer() as elapsed:
        xests_v0 = omp_v0(y, X, X.T @ X, (X.T @ y.T[:, :, None]).squeeze(-1), n_nonzero_coefs)
    print('Samples per second:', n_samples/elapsed())
    print("\n")
    exit(0)

    print('Single core. Naive Implementation, based on our Homework.')
    with elapsed_timer() as elapsed:
        xests_naive = omp_naive(X, y, n_nonzero_coefs)
    print('Samples per second:', n_samples/elapsed())
    print("\n")

    exit(0)
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
    err_sklearn = np.linalg.norm(y.T - (X @ omp.coef_[:, :, None]).squeeze(-1), 2, 1)
    avg_ylen = np.linalg.norm(y, 2, 0)
    # print(np.median(naive_err) / avg_ylen, np.median(scipy_err) / avg_ylen)
    plt.plot(np.sort(err_naive / avg_ylen), label='Naive')
    plt.plot(np.sort(err_v0 / avg_ylen), '.', label='v0')
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

