import os
import sys
import torch
import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import OrthogonalMatchingPursuit
from contextlib import contextmanager
from timeit import default_timer

sys.path.append(os.path.join(os.path.dirname(__file__), "cython"))

from test_omp import omp_naive
from cython.test import *

from main import run_omp, elapsed_timer

BENCHMARKS = {
    'image_patches': {
        'n_features': 256,
        'n_components': 1024,
        'n_nonzero_coefs': 32,
        'n_samples': 5000,
    },
    'face_recognition': {
        'n_features': 8064,
        'n_components': 1207,
        'n_nonzero_coefs': 30,
        'n_samples': 1207,
    },
    'audio': {
        'n_features': 512,
        'n_components': 2048,
        'n_nonzero_coefs': 64,
        'n_samples': 5000,
    },
}


def run_benchmark(name, cfg):
    n_features = cfg['n_features']
    n_components = cfg['n_components']
    n_nonzero_coefs = cfg['n_nonzero_coefs']
    n_samples = cfg['n_samples']

    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"  n_features={n_features}, n_components={n_components}, "
          f"n_nonzero_coefs={n_nonzero_coefs}, n_samples={n_samples}")
    print(f"{'='*60}")

    y, X, w = make_sparse_coded_signal(
        n_samples=n_samples,
        n_components=n_components,
        n_features=n_features,
        n_nonzero_coefs=n_nonzero_coefs,
        random_state=0,
    )
    # new sklearn returns: y (n_samples, n_features), X (n_components, n_features)
    # run_omp expects: X (n_features, n_components), y (n_samples, n_features)
    X = X.T

    results = {}

    with elapsed_timer() as elapsed:
        omp = run_omp(X.copy(), y.copy(), n_nonzero_coefs,
                      tol=None, normalize=False, fit_intercept=False, alg='sklearn')
    t_sklearn = elapsed()
    results['sklearn'] = {'time': t_sklearn, 'sps': n_samples / t_sklearn, 'coefs': omp.coef_}
    print(f"sklearn:  {results['sklearn']['sps']:.0f} samples/sec ({t_sklearn:.2f}s)")

    with elapsed_timer() as elapsed:
        xests_naive = run_omp(X.copy(), y.copy(), n_nonzero_coefs,
                              tol=None, normalize=False, fit_intercept=False, alg='naive')
    t_naive = elapsed()
    results['naive'] = {'time': t_naive, 'sps': n_samples / t_naive, 'coefs': xests_naive.numpy()}
    print(f"naive:    {results['naive']['sps']:.0f} samples/sec ({t_naive:.2f}s)")

    with elapsed_timer() as elapsed:
        xests_v0 = run_omp(torch.as_tensor(X.copy()), torch.as_tensor(y.copy()), n_nonzero_coefs,
                           tol=None, normalize=False, fit_intercept=False, alg='v0')
    t_v0 = elapsed()
    results['v0'] = {'time': t_v0, 'sps': n_samples / t_v0, 'coefs': xests_v0.numpy()}
    print(f"v0:       {results['v0']['sps']:.0f} samples/sec ({t_v0:.2f}s)")

    print(f"\nSpeedups vs sklearn:")
    print(f"  naive: {results['naive']['sps'] / results['sklearn']['sps']:.2f}x")
    print(f"  v0:    {results['v0']['sps'] / results['sklearn']['sps']:.2f}x")

    # Correctness checks
    print(f"\nCorrectness checks:")
    eps = 1e-12
    A = results['sklearn']['coefs']
    B = results['naive']['coefs']
    C = results['v0']['coefs']

    # naive vs v0 support agreement
    support_diffs = 0
    for i in range(B.shape[0]):
        nzB = np.flatnonzero(np.abs(B[i]) > eps).tolist()
        nzC = np.flatnonzero(np.abs(C[i]) > eps).tolist()
        if not np.array_equal(nzB, nzC):
            support_diffs += 1
    if support_diffs:
        print(f"  WARNING: {support_diffs}/{B.shape[0]} samples have naive vs v0 support disagreement")
    else:
        print(f"  naive vs v0 support: agree on all {B.shape[0]} samples")

    # Residuals (no normalization/centering since normalize=False, fit_intercept=False)
    r_sklearn = y - (X @ A.T).T
    r_v0 = y - (X @ C.T).T
    r_naive = y - (X @ B.T).T

    for label, resid in [('sklearn', r_sklearn), ('v0', r_v0), ('naive', r_naive)]:
        print(f"  Max reconstruction error ({label}): {(resid ** 2).sum(axis=1).max():.6e}")

    # Orthogonality check
    for label, coefs, resid in [('sklearn', A, r_sklearn), ('v0', C, r_v0), ('naive', B, r_naive)]:
        orth_violations = []
        for i in range(coefs.shape[0]):
            nz = np.flatnonzero(np.abs(coefs[i]) > eps)
            if len(nz) > 0:
                orth_violations.append(np.abs(X[:, nz].T @ resid[i]).max())
        print(f"  Max orthogonality violation ({label}): {max(orth_violations):.2e}" if orth_violations else f"  Max orthogonality violation ({label}): 0")

    return results


if __name__ == '__main__':
    selected = sys.argv[1:] if len(sys.argv) > 1 else BENCHMARKS.keys()
    for name in selected:
        if name not in BENCHMARKS:
            print(f"Unknown benchmark: {name}. Available: {', '.join(BENCHMARKS.keys())}")
            continue
        run_benchmark(name, BENCHMARKS[name])
