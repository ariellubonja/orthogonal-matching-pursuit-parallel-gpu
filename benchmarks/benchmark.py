import os
import sys
import torch
import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import OrthogonalMatchingPursuit
from datetime import datetime

from batched_omp import run_omp, elapsed_timer
from batched_omp.blas_kernels import argmax_blast, update_projections_blast, update_D_mybest_blast


def omp_v0_blas(X_np, y_np, n_nonzero_coefs):
    """v0 using Cython BLAS wrappers for inner-loop ops (argmax, D_mybest update, projection update)."""
    B, N = y_np.shape
    M = X_np.shape[1]  # n_components
    XTX_np = (X_np.T @ X_np)  # (M, M)

    # Initial projections: projections[b, m] = X[:, m]^T @ y[b]
    projections = (y_np @ X_np).copy()                 # (B, M)
    sets = np.zeros((n_nonzero_coefs, B), dtype=np.int64)
    F = np.eye(n_nonzero_coefs, dtype=np.float64)[None].repeat(B, axis=0)   # (B, K, K)
    a_F = np.zeros((n_nonzero_coefs, B, 1), dtype=np.float64)
    D_mybest = np.empty((B, n_nonzero_coefs, M), dtype=np.float64)
    temp_F_k_k = np.ones((B,), dtype=np.float64)

    arange_B = np.arange(B, dtype=np.int64)

    for k in range(n_nonzero_coefs):
        argmax_blast(np.abs(projections), sets[k])                            # sets[k] = argmax |proj|

        D_mybest[:, k, :] = XTX_np[sets[k], :]                               # gather rows of XTX

        if k:
            # D_mybest_maxindices[i] = column sets[k,i] of D_mybest[i, :k, :]
            D_mybest_maxindices = D_mybest[:, :k, :].transpose(0, 2, 1)[
                arange_B, sets[k], :]                                         # (B, k)

            temp_F_k_k[:] = 1.0 / np.sqrt(1.0 - (D_mybest_maxindices ** 2).sum(axis=1))

            # fused: D_mybest[:,k,:] = temp_F_k_k * (D_mybest[:,k,:] - D_mybest[:,:k,:]^T @ D_mybest_maxindices)
            update_D_mybest_blast(temp_F_k_k, XTX_np, sets[k],
                                  D_mybest[:, :k, :],
                                  D_mybest_maxindices,
                                  D_mybest[:, k, :])
        else:
            temp_F_k_k[:] = 1.0

        temp_a_F = temp_F_k_k * projections[arange_B, sets[k]]               # (B,)
        update_projections_blast(projections, D_mybest[:, k, :], -temp_a_F)  # projections -= temp_a_F * D[:,k,:]

        a_F[k, :, 0] = temp_a_F
        if k:
            F[:, k, :k] = (D_mybest_maxindices[:, None, :] @
                           F[:, :k, :k]).squeeze(1) * (-temp_F_k_k[:, None])
            F[:, k, k] = temp_F_k_k

    solutions = (F.transpose(0, 2, 1) @
                 a_F.squeeze(-1).T[:, :, None])         # (B, K, 1)
    return sets.T, solutions


def run_sklearn(X, y, n_nonzero_coefs, tol=None):
    omp_args = dict(tol=tol, n_nonzero_coefs=n_nonzero_coefs, precompute='auto', fit_intercept=False)
    omp = OrthogonalMatchingPursuit(**omp_args)
    omp.fit(X, y.T)
    return omp


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

# Paper's Fig 1: A ∈ R^(M×8M), y ∈ R^(B×M), S=M/4, B=100
# M is the variable, ranging from 16 to 2048
PAPER_M_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]

HAS_CUDA = torch.cuda.is_available()


def gpu_warmup():
    if not HAS_CUDA:
        return
    x = torch.randn(100, 100, device='cuda')
    _ = x @ x.T
    torch.cuda.synchronize()


def run_benchmark(name, cfg, run_gpu=True):
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

    # --- CPU benchmarks ---
    with elapsed_timer() as elapsed:
        omp = run_sklearn(X.copy(), y.copy(), n_nonzero_coefs, tol=None)
    t = elapsed()
    results['sklearn'] = {'time': t, 'sps': n_samples / t, 'coefs': omp.coef_}
    print(f"CPU sklearn:  {results['sklearn']['sps']:>10.0f} samples/sec ({t:.3f}s)")

    with elapsed_timer() as elapsed:
        xests_naive = run_omp(X.copy(), y.copy(), n_nonzero_coefs,
                              tol=None, normalize=False, fit_intercept=False, alg='naive')
    t = elapsed()
    results['naive_cpu'] = {'time': t, 'sps': n_samples / t, 'coefs': xests_naive.numpy()}
    print(f"CPU naive:    {results['naive_cpu']['sps']:>10.0f} samples/sec ({t:.3f}s)")

    with elapsed_timer() as elapsed:
        xests_v0 = run_omp(torch.as_tensor(X.copy()), torch.as_tensor(y.copy()), n_nonzero_coefs,
                           tol=None, normalize=False, fit_intercept=False, alg='v0')
    t = elapsed()
    results['v0_cpu'] = {'time': t, 'sps': n_samples / t, 'coefs': xests_v0.numpy()}
    print(f"CPU v0:       {results['v0_cpu']['sps']:>10.0f} samples/sec ({t:.3f}s)")

    with elapsed_timer() as elapsed:
        sets_blas, sols_blas = omp_v0_blas(X.copy(), y.copy(), n_nonzero_coefs)
    t = elapsed()
    # reconstruct dense coef matrix from (sets, solutions)
    xests_blas = np.zeros((n_samples, n_components))
    for i in range(n_samples):
        xests_blas[i, sets_blas[i]] = sols_blas[i, :, 0]
    results['v0_blas'] = {'time': t, 'sps': n_samples / t, 'coefs': xests_blas}
    print(f"CPU v0 blas:  {results['v0_blas']['sps']:>10.0f} samples/sec ({t:.3f}s)")

    # --- GPU benchmarks ---
    if run_gpu and HAS_CUDA:
        X_cuda = torch.as_tensor(X.copy()).cuda()
        y_cuda = torch.as_tensor(y.copy()).cuda()

        for alg in ['naive', 'v0']:
            key = f'{alg}_gpu'
            try:
                torch.cuda.synchronize()
                with elapsed_timer() as elapsed:
                    xests_gpu = run_omp(X_cuda.clone(), y_cuda.clone(), n_nonzero_coefs,
                                        tol=None, normalize=False, fit_intercept=False, alg=alg)
                    torch.cuda.synchronize()
                t = elapsed()
                results[key] = {'time': t, 'sps': n_samples / t, 'coefs': xests_gpu.cpu().numpy()}
                print(f"GPU {alg + ':':9s} {results[key]['sps']:>10.0f} samples/sec ({t:.3f}s)")
            except torch.cuda.OutOfMemoryError:
                print(f"GPU {alg + ':':9s}        OOM")
                torch.cuda.empty_cache()

    # --- Speedups ---
    sklearn_sps = results['sklearn']['sps']
    print(f"\nSpeedups vs sklearn:")
    for key in ['naive_cpu', 'v0_cpu', 'v0_blas', 'naive_gpu', 'v0_gpu']:
        if key in results:
            label = key.replace('_', ' ').upper()
            print(f"  {label}: {results[key]['sps'] / sklearn_sps:.1f}x")

    # --- Correctness checks ---
    print(f"\nCorrectness:")
    eps = 1e-12
    coefs_map = {k: v['coefs'] for k, v in results.items()}

    # Check naive vs v0 support agreement (CPU versions)
    B = coefs_map['naive_cpu']
    C = coefs_map['v0_cpu']
    support_diffs = sum(1 for i in range(B.shape[0])
                        if not np.array_equal(
                            np.flatnonzero(np.abs(B[i]) > eps),
                            np.flatnonzero(np.abs(C[i]) > eps)))
    if support_diffs:
        print(f"  WARNING: {support_diffs}/{B.shape[0]} samples have CPU naive vs v0 support disagreement")
    else:
        print(f"  CPU naive vs v0 support: agree on all {B.shape[0]} samples")

    # Check GPU vs CPU agreement if GPU was run
    if 'v0_gpu' in coefs_map:
        C_gpu = coefs_map['v0_gpu']
        gpu_diffs = sum(1 for i in range(C.shape[0])
                        if not np.array_equal(
                            np.flatnonzero(np.abs(C[i]) > eps),
                            np.flatnonzero(np.abs(C_gpu[i]) > eps)))
        if gpu_diffs:
            print(f"  WARNING: {gpu_diffs}/{C.shape[0]} samples have v0 CPU vs GPU support disagreement")
        else:
            print(f"  v0 CPU vs GPU support: agree on all {C.shape[0]} samples")

    # Check v0_blas vs v0_cpu support agreement
    if 'v0_blas' in coefs_map:
        C_blas = coefs_map['v0_blas']
        blas_diffs = sum(1 for i in range(C.shape[0])
                         if not np.array_equal(
                             np.flatnonzero(np.abs(C[i]) > eps),
                             np.flatnonzero(np.abs(C_blas[i]) > eps)))
        if blas_diffs:
            print(f"  WARNING: {blas_diffs}/{C.shape[0]} samples have v0_cpu vs v0_blas support disagreement")
        else:
            print(f"  v0 CPU vs v0 BLAS support: agree on all {C.shape[0]} samples")

    # Orthogonality check (CPU v0 and GPU v0)
    for key in ['v0_cpu', 'v0_blas', 'v0_gpu']:
        if key not in coefs_map:
            continue
        coefs = coefs_map[key]
        resid = y - (X @ coefs.T).T
        orth_violations = []
        for i in range(coefs.shape[0]):
            nz = np.flatnonzero(np.abs(coefs[i]) > eps)
            if len(nz) > 0:
                orth_violations.append(np.abs(X[:, nz].T @ resid[i]).max())
        label = key.replace('_', ' ')
        print(f"  Max orthogonality violation ({label}): {max(orth_violations):.2e}" if orth_violations else f"  Max orthogonality violation ({label}): 0")

    return results


def run_paper_benchmarks(run_gpu=True):
    """Reproduce Fig 1 from paper: A ∈ R^(M×N), N=8M, S=M/4, B=100"""
    print(f"\n{'#'*60}")
    print(f"Paper Fig 1 benchmarks: N=8M, S=M/4, B=100")
    print(f"{'#'*60}")

    all_results = {}
    for M in PAPER_M_VALUES:
        cfg = {
            'n_features': M,
            'n_components': 8 * M,
            'n_nonzero_coefs': M // 4,
            'n_samples': 100,
        }
        if cfg['n_nonzero_coefs'] < 1:
            cfg['n_nonzero_coefs'] = 1
        all_results[M] = run_benchmark(f"paper_M={M}", cfg, run_gpu=run_gpu)

    # Summary table
    print(f"\n{'='*60}")
    print("Paper Fig 1 Summary (time in seconds)")
    print(f"{'='*60}")
    header = f"{'M':>6} | {'sklearn':>8} | {'naive':>8} | {'v0 CPU':>8}"
    if run_gpu and HAS_CUDA:
        header += f" | {'naive GPU':>9} | {'v0 GPU':>8}"
    print(header)
    print("-" * len(header))
    for M, res in all_results.items():
        row = f"{M:>6} | {res['sklearn']['time']:>8.3f} | {res['naive_cpu']['time']:>8.3f} | {res['v0_cpu']['time']:>8.3f}"
        if run_gpu and HAS_CUDA:
            naive_gpu = f"{res['naive_gpu']['time']:>9.3f}" if 'naive_gpu' in res else "      OOM"
            v0_gpu = f"{res['v0_gpu']['time']:>8.3f}" if 'v0_gpu' in res else "     OOM"
            row += f" | {naive_gpu} | {v0_gpu}"
        print(row)

    return all_results


class Tee:
    """Write to both stdout and a file."""
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        if not self.file.closed:
            self.file.flush()


if __name__ == '__main__':
    gpu_warmup()

    args = sys.argv[1:]
    no_gpu = '--no-gpu' in args
    args = [a for a in args if a != '--no-gpu']

    run_gpu = HAS_CUDA and not no_gpu

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(results_dir, f'benchmark_{timestamp}.txt')

    with open(results_path, 'w') as f:
        tee = Tee(f, sys.stdout)
        sys.stdout = tee

        cpu_name = "unknown"
        try:
            with open('/proc/cpuinfo') as cpuinfo:
                for line in cpuinfo:
                    if line.startswith('model name'):
                        cpu_name = line.split(':')[1].strip()
                        break
        except OSError:
            pass
        gpu_name = torch.cuda.get_device_name(0) if HAS_CUDA else "N/A"
        print(f"Date: {datetime.now().isoformat()}")
        print(f"CPU: {cpu_name}")
        print(f"GPU: {gpu_name}")

        if not args or 'all' in args:
            for name in BENCHMARKS:
                run_benchmark(name, BENCHMARKS[name], run_gpu=run_gpu)
            run_paper_benchmarks(run_gpu=run_gpu)
        else:
            for name in args:
                if name == 'paper':
                    run_paper_benchmarks(run_gpu=run_gpu)
                elif name in BENCHMARKS:
                    run_benchmark(name, BENCHMARKS[name], run_gpu=run_gpu)
                else:
                    print(f"Unknown benchmark: {name}. Available: {', '.join(BENCHMARKS.keys())}, paper, all")

        sys.stdout = tee.stream

    print(f"\nResults written to {results_path}")
