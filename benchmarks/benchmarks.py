import os
import sys
import torch
import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import OrthogonalMatchingPursuit
from datetime import datetime

from batched_omp import run_omp, omp_v0, omp_naive, omp_v0_blas, elapsed_timer

try:
    import spams
    HAS_SPAMS = True
except ImportError:
    HAS_SPAMS = False

try:
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'  # limit JAX to 40% GPU memory
    import jax
    import jax.numpy as jnp
    import cr.sparse.pursuit.omp as cr_omp
    HAS_CR_SPARSE = True
    _cr_sparse_jit_cache = {}
except ImportError:
    HAS_CR_SPARSE = False


def run_sklearn(X, y, n_nonzero_coefs, tol=None):
    omp_args = dict(tol=tol, n_nonzero_coefs=n_nonzero_coefs, precompute='auto', fit_intercept=False)
    omp = OrthogonalMatchingPursuit(**omp_args)
    omp.fit(X, y.T)
    return omp


def run_cr_sparse(X, y, n_nonzero_coefs):
    """Run cr-sparse OMP via JAX vmap. X: (n_features, n_components), y: (n_samples, n_features)."""
    Phi = jnp.array(X, dtype=jnp.float32)
    Y = jnp.array(y, dtype=jnp.float32)
    # Cache the JIT-compiled batched solve per sparsity level
    if n_nonzero_coefs not in _cr_sparse_jit_cache:
        solve_fn = lambda yi: cr_omp.matrix_solve(Phi, yi, max_iters=n_nonzero_coefs).x
        _cr_sparse_jit_cache[n_nonzero_coefs] = jax.vmap(solve_fn)
    solve_batch = _cr_sparse_jit_cache[n_nonzero_coefs]
    coefs = solve_batch(Y)
    coefs.block_until_ready()
    return np.array(coefs)


def run_spams(X, y, n_nonzero_coefs):
    """Run SPAMS OMP. X: (n_features, n_components), y: (n_samples, n_features)."""
    D = np.asfortranarray(X, dtype=np.float64)
    signals = np.asfortranarray(y.T, dtype=np.float64)
    alpha = spams.omp(signals, D, L=n_nonzero_coefs, eps=0.0, numThreads=-1)
    return np.asarray(alpha.todense()).T


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


def spams_warmup():
    if not HAS_SPAMS:
        return
    D = np.asfortranarray(np.eye(4, dtype=np.float64))
    x = np.asfortranarray(np.ones((4, 1), dtype=np.float64))
    spams.omp(x, D, L=1, numThreads=-1)


def run_benchmark(name, cfg, run_gpu=True, skip_correctness=False, skip_sklearn=False):
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
    if not skip_sklearn:
        with elapsed_timer() as elapsed:
            omp = run_sklearn(X.copy(), y.copy(), n_nonzero_coefs, tol=None)
        t = elapsed()
        results['sklearn'] = {'time': t, 'sps': n_samples / t, 'coefs': omp.coef_}
        print(f"CPU sklearn:  {results['sklearn']['sps']:>10.0f} samples/sec ({t:.3f}s)")

    if HAS_SPAMS:
        for _ in range(3):  # warmup (OpenMP thread pool needs several calls to stabilize)
            run_spams(X.copy(), y.copy(), n_nonzero_coefs)
        spams_times = []
        for _ in range(3):
            with elapsed_timer() as elapsed:
                spams_coefs = run_spams(X.copy(), y.copy(), n_nonzero_coefs)
            spams_times.append(elapsed())
        t = np.mean(spams_times)
        t_std = np.std(spams_times)
        results['spams'] = {'time': t, 'time_std': t_std, 'sps': n_samples / t, 'coefs': spams_coefs}
        print(f"CPU SPAMS:    {results['spams']['sps']:>10.0f} samples/sec ({t:.3f}s +/- {t_std:.3f}s)")

    if HAS_CR_SPARSE:
        try:
            run_cr_sparse(X.copy(), y.copy(), n_nonzero_coefs)  # warmup (JIT compile)
            with elapsed_timer() as elapsed:
                cr_coefs = run_cr_sparse(X.copy(), y.copy(), n_nonzero_coefs)
            t = elapsed()
            results['cr_sparse'] = {'time': t, 'sps': n_samples / t, 'coefs': cr_coefs}
            print(f"GPU cr-sparse:{results['cr_sparse']['sps']:>10.0f} samples/sec ({t:.3f}s)")
        except Exception as e:
            print(f"GPU cr-sparse:       FAIL ({e})")

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
    if 'sklearn' in results:
        sklearn_sps = results['sklearn']['sps']
        print(f"\nSpeedups vs sklearn:")
        for key in ['spams', 'cr_sparse', 'naive_cpu', 'v0_cpu', 'v0_blas', 'naive_gpu', 'v0_gpu']:
            if key in results:
                label = key.replace('_', ' ').upper()
                print(f"  {label}: {results[key]['sps'] / sklearn_sps:.1f}x")

    # --- Correctness checks ---
    if skip_correctness:
        return results
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

    # Check SPAMS vs v0_cpu support agreement
    if 'spams' in coefs_map:
        C_spams = coefs_map['spams']
        spams_diffs = sum(1 for i in range(C.shape[0])
                          if not np.array_equal(
                              np.flatnonzero(np.abs(C_spams[i]) > eps),
                              np.flatnonzero(np.abs(C[i]) > eps)))
        if spams_diffs:
            print(f"  SPAMS vs v0 CPU support: {spams_diffs}/{C.shape[0]} samples disagree (expected — different tie-breaking)")
        else:
            print(f"  SPAMS vs v0 CPU support: agree on all {C.shape[0]} samples")

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
    for key in ['spams', 'cr_sparse', 'v0_cpu', 'v0_blas', 'v0_gpu']:
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
    header = f"{'M':>6} | {'sklearn':>8} | {'SPAMS':>8} | {'naive':>8} | {'v0 CPU':>8}"
    if run_gpu and HAS_CUDA:
        header += f" | {'naive GPU':>9} | {'v0 GPU':>8}"
    print(header)
    print("-" * len(header))
    for M, res in all_results.items():
        spams_col = f"{res['spams']['time']:>8.3f}" if 'spams' in res else "     N/A"
        row = f"{M:>6} | {res['sklearn']['time']:>8.3f} | {spams_col} | {res['naive_cpu']['time']:>8.3f} | {res['v0_cpu']['time']:>8.3f}"
        if run_gpu and HAS_CUDA:
            naive_gpu = f"{res['naive_gpu']['time']:>9.3f}" if 'naive_gpu' in res else "      OOM"
            v0_gpu = f"{res['v0_gpu']['time']:>8.3f}" if 'v0_gpu' in res else "     OOM"
            row += f" | {naive_gpu} | {v0_gpu}"
        print(row)

    return all_results


SWEEP_N_VALUES = [64, 128, 256, 512, 1024, 2048, 4096]
SWEEP_B_VALUES = [10, 50, 100, 500, 1000, 5000]
SWEEP_S_VALUES = [8, 32, 64]


def run_sweep(run_gpu=True):
    """2D parameter sweep: N (n_components) x B (n_samples) for each sparsity level S."""
    import json
    from timeit import default_timer

    # Build list of valid cells (S must be <= M = N/4)
    cells = []
    for S in SWEEP_S_VALUES:
        for N in SWEEP_N_VALUES:
            M = N // 4
            if S > M:
                continue
            for B in SWEEP_B_VALUES:
                cells.append((S, N, B))

    total = len(cells)
    print(f"\n{'#'*60}")
    print(f"Parameter sweep: {total} cells")
    print(f"  N (n_components): {SWEEP_N_VALUES}")
    print(f"  B (n_samples):    {SWEEP_B_VALUES}")
    print(f"  S (sparsity):     {SWEEP_S_VALUES}")
    print(f"  Fixed: M = N/4")
    print(f"{'#'*60}")

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

    sweep_results = {
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'cpu': cpu_name,
            'gpu': gpu_name,
            'N_values': SWEEP_N_VALUES,
            'B_values': SWEEP_B_VALUES,
            'S_values': SWEEP_S_VALUES,
        },
        'cells': {},
    }

    start_time = default_timer()

    for cell_num, (S, N, B) in enumerate(cells):
        M = N // 4
        elapsed = default_timer() - start_time
        if cell_num > 0:
            eta = elapsed / cell_num * (total - cell_num)
            eta_str = f"ETA: {eta / 60:.1f}min"
        else:
            eta_str = "ETA: --"
        print(f"\n[{cell_num + 1}/{total}] N={N}, B={B}, S={S} | elapsed: {elapsed / 60:.1f}min, {eta_str}")

        cfg = {
            'n_features': M,
            'n_components': N,
            'n_nonzero_coefs': S,
            'n_samples': B,
        }
        results = run_benchmark(f"sweep_N{N}_B{B}_S{S}", cfg,
                                run_gpu=run_gpu, skip_correctness=True,
                                skip_sklearn=True)

        # Strip coefs (not JSON-serializable, not needed for plotting)
        stripped = {}
        for alg, data in results.items():
            if isinstance(data, dict) and 'coefs' in data:
                entry = {'time': data['time'], 'sps': data['sps']}
                if 'time_std' in data:
                    entry['time_std'] = data['time_std']
                stripped[alg] = entry
            else:
                stripped[alg] = data

        # Mark missing GPU algorithms as OOM
        if run_gpu and HAS_CUDA:
            for alg_key in ['naive_gpu', 'v0_gpu']:
                if alg_key not in stripped:
                    stripped[alg_key] = 'OOM'
        if HAS_CR_SPARSE and 'cr_sparse' not in stripped:
            stripped['cr_sparse'] = 'FAIL'

        sweep_results['cells'][f"{S}_{N}_{B}"] = stripped

    total_time = default_timer() - start_time
    print(f"\n{'='*60}")
    print(f"Sweep complete: {total} cells in {total_time / 60:.1f} minutes")
    print(f"{'='*60}")

    # Save JSON
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(results_dir, f'sweep_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"Sweep results saved to {json_path}")

    return sweep_results


# ── Ablation study ──────────────────────────────────────────────

ABLATION_CONFIGS = {
    'small':  {'n_features': 32,   'n_components': 128,  'n_nonzero_coefs': 8,  'n_samples': 500},
    'medium': {'n_features': 128,  'n_components': 512,  'n_nonzero_coefs': 32, 'n_samples': 1000},
    'large':  {'n_features': 512,  'n_components': 2048, 'n_nonzero_coefs': 64, 'n_samples': 5000},
    'face':   {'n_features': 8064, 'n_components': 1207, 'n_nonzero_coefs': 30, 'n_samples': 1207},
}


def _ablation_a1(X_t, y_t, XTX, S, B, cell, device='cpu'):
    """A1: Batching — v0 full batch vs single-sample loop."""
    is_gpu = device == 'gpu'
    suffix = f'_{device}'

    # Warmup (compiles GPU kernels, stabilizes CPU caches)
    omp_v0(X_t, y_t[:1], XTX, n_nonzero_coefs=S)
    if is_gpu:
        torch.cuda.synchronize()

    if is_gpu:
        torch.cuda.synchronize()
    with elapsed_timer() as elapsed:
        omp_v0(X_t, y_t, XTX, n_nonzero_coefs=S)
        if is_gpu:
            torch.cuda.synchronize()
    cell[f'a1_batched{suffix}'] = elapsed()
    print(f"  v0 batched {device.upper():<3}:           {cell[f'a1_batched{suffix}']:.3f}s")

    if is_gpu:
        torch.cuda.synchronize()
    with elapsed_timer() as elapsed:
        for i in range(B):
            omp_v0(X_t, y_t[i:i+1], XTX, n_nonzero_coefs=S)
        if is_gpu:
            torch.cuda.synchronize()
    cell[f'a1_loop{suffix}'] = elapsed()
    spd = cell[f'a1_loop{suffix}'] / cell[f'a1_batched{suffix}']
    print(f"  v0 loop (B=1) {device.upper():<3}:        {cell[f'a1_loop{suffix}']:.3f}s  ({spd:.1f}x slower)")


def _ablation_a2(X_t, y_t, XTX, S, cell, device='cpu'):
    """A2: Gram precomputation — naive with vs without precomputed X^T X."""
    is_gpu = device == 'gpu'
    suffix = f'_{device}'

    # Warmup
    omp_naive(X_t, y_t[:1], 1, XTX=XTX)
    if is_gpu:
        torch.cuda.synchronize()

    if is_gpu:
        torch.cuda.synchronize()
    with elapsed_timer() as elapsed:
        omp_naive(X_t, y_t, S, XTX=XTX)
        if is_gpu:
            torch.cuda.synchronize()
    cell[f'a2_precompute{suffix}'] = elapsed()
    print(f"  naive + precompute {device.upper():<3}:    {cell[f'a2_precompute{suffix}']:.3f}s")

    if is_gpu:
        torch.cuda.synchronize()
    with elapsed_timer() as elapsed:
        omp_naive(X_t, y_t, S, XTX=None)
        if is_gpu:
            torch.cuda.synchronize()
    cell[f'a2_no_precompute{suffix}'] = elapsed()
    spd = cell[f'a2_no_precompute{suffix}'] / cell[f'a2_precompute{suffix}']
    print(f"  naive no precomp {device.upper():<3}:      {cell[f'a2_no_precompute{suffix}']:.3f}s  ({spd:.1f}x slower)")


def _ablation_a3(X_t, y_t, XTX, S, cell, device='cpu'):
    """A3: Inverse Cholesky — v0 with iterative inverse vs standard Cholesky solve."""
    is_gpu = device == 'gpu'
    suffix = f'_{device}'

    # Warmup
    omp_v0(X_t, y_t[:1], XTX, n_nonzero_coefs=S)
    if is_gpu:
        torch.cuda.synchronize()

    if is_gpu:
        torch.cuda.synchronize()
    with elapsed_timer() as elapsed:
        omp_v0(X_t, y_t, XTX, n_nonzero_coefs=S, inverse_cholesky=True)
        if is_gpu:
            torch.cuda.synchronize()
    cell[f'a3_inv_chol{suffix}'] = elapsed()
    print(f"  v0 inv Cholesky {device.upper():<3}:      {cell[f'a3_inv_chol{suffix}']:.3f}s")

    if is_gpu:
        torch.cuda.synchronize()
    with elapsed_timer() as elapsed:
        omp_v0(X_t, y_t, XTX, n_nonzero_coefs=S, inverse_cholesky=False)
        if is_gpu:
            torch.cuda.synchronize()
    cell[f'a3_std_chol{suffix}'] = elapsed()
    spd = cell[f'a3_std_chol{suffix}'] / cell[f'a3_inv_chol{suffix}']
    print(f"  v0 std Cholesky {device.upper():<3}:      {cell[f'a3_std_chol{suffix}']:.3f}s  ({spd:.1f}x slower)")


def run_ablation(run_gpu=True):
    """Ablation study: isolate the contribution of batching, Gram precomputation, and inverse Cholesky."""
    import json

    all_results = {}

    for config_name, cfg in ABLATION_CONFIGS.items():
        M = cfg['n_features']
        N = cfg['n_components']
        S = cfg['n_nonzero_coefs']
        B = cfg['n_samples']

        print(f"\n{'='*60}")
        print(f"Ablation: {config_name}  (M={M}, N={N}, S={S}, B={B})")
        print(f"{'='*60}")

        y, X, w = make_sparse_coded_signal(
            n_samples=B, n_components=N, n_features=M,
            n_nonzero_coefs=S, random_state=0,
        )
        X = X.T  # (M, N)

        X_t = torch.as_tensor(X, dtype=torch.float64)
        y_t = torch.as_tensor(y, dtype=torch.float64)
        XTX = X_t.T @ X_t

        cell = {}

        # Reference: sklearn
        with elapsed_timer() as elapsed:
            run_sklearn(X.copy(), y.copy(), S)
        cell['sklearn'] = elapsed()
        print(f"  sklearn (reference):       {cell['sklearn']:.3f}s")

        # --- CPU ablations ---
        print(f"\n  --- A1: Batching (CPU) ---")
        _ablation_a1(X_t, y_t, XTX, S, B, cell, device='cpu')

        print(f"\n  --- A2: Gram precomputation (CPU) ---")
        _ablation_a2(X_t, y_t, XTX, S, cell, device='cpu')

        print(f"\n  --- A3: Inverse Cholesky (CPU) ---")
        _ablation_a3(X_t, y_t, XTX, S, cell, device='cpu')

        # --- GPU ablations ---
        if run_gpu and HAS_CUDA:
            X_cuda = X_t.cuda()
            y_cuda = y_t.cuda()
            XTX_cuda = XTX.cuda()

            for ablation_name, ablation_fn, needs_B in [
                ('A1: Batching (GPU)', _ablation_a1, True),
                ('A2: Gram precomputation (GPU)', _ablation_a2, False),
                ('A3: Inverse Cholesky (GPU)', _ablation_a3, False),
            ]:
                print(f"\n  --- {ablation_name} ---")
                try:
                    if needs_B:
                        ablation_fn(X_cuda, y_cuda, XTX_cuda, S, B, cell, device='gpu')
                    else:
                        ablation_fn(X_cuda, y_cuda, XTX_cuda, S, cell, device='gpu')
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM")
                    torch.cuda.empty_cache()

        all_results[config_name] = cell

    # --- Summary table ---
    print(f"\n{'#'*60}")
    print("Ablation Summary — Speedup from each optimization (CPU)")
    print(f"{'#'*60}")
    print(f"  {'Config':<10} | {'A1 Batch':>10} | {'A2 Gram':>10} | {'A3 InvChol':>10}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for config_name, cell in all_results.items():
        a1 = cell.get('a1_loop_cpu', 0) / max(cell.get('a1_batched_cpu', 1), 1e-9)
        a2 = cell.get('a2_no_precompute_cpu', 0) / max(cell.get('a2_precompute_cpu', 1), 1e-9)
        a3 = cell.get('a3_std_chol_cpu', 0) / max(cell.get('a3_inv_chol_cpu', 1), 1e-9)
        print(f"  {config_name:<10} | {a1:>9.1f}x | {a2:>9.1f}x | {a3:>9.1f}x")

    if run_gpu and HAS_CUDA:
        print(f"\nAblation Summary — Speedup from each optimization (GPU)")
        print(f"  {'Config':<10} | {'A1 Batch':>10} | {'A2 Gram':>10} | {'A3 InvChol':>10}")
        print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        for config_name, cell in all_results.items():
            a1 = cell.get('a1_loop_gpu', 0) / max(cell.get('a1_batched_gpu', 1), 1e-9)
            a2 = cell.get('a2_no_precompute_gpu', 0) / max(cell.get('a2_precompute_gpu', 1), 1e-9)
            a3 = cell.get('a3_std_chol_gpu', 0) / max(cell.get('a3_inv_chol_gpu', 1), 1e-9)
            vals = []
            for v in [a1, a2, a3]:
                vals.append(f'{v:>9.1f}x' if v > 0 else f'{"OOM":>10}')
            print(f"  {config_name:<10} | {vals[0]} | {vals[1]} | {vals[2]}")

    # Save JSON
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(results_dir, f'ablation_{timestamp}.json')

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

    output = {
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'cpu': cpu_name,
            'gpu': gpu_name,
            'configs': ABLATION_CONFIGS,
        },
        'results': all_results,
    }
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nAblation results saved to {json_path}")

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
    spams_warmup()

    args = sys.argv[1:]
    no_gpu = '--no-gpu' in args
    args = [a for a in args if not a.startswith('--')]

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
                if name == 'sweep':
                    run_sweep(run_gpu=run_gpu)
                elif name == 'paper':
                    run_paper_benchmarks(run_gpu=run_gpu)
                elif name == 'ablation':
                    run_ablation(run_gpu=run_gpu)
                elif name in BENCHMARKS:
                    run_benchmark(name, BENCHMARKS[name], run_gpu=run_gpu)
                else:
                    print(f"Unknown benchmark: {name}. Available: {', '.join(BENCHMARKS.keys())}, paper, sweep, ablation, all")

        sys.stdout = tee.stream

    print(f"\nResults written to {results_path}")
