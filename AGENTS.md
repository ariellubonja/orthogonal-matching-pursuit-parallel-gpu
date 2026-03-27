# Batched OMP — Agent Guide

Paper: https://arxiv.org/abs/2407.06434
By Ariel Lubonja, Sebastian Kazmarek Praesius, and Trac Duy Tran.

## Build & test

```bash
pip install -e ".[dev]"        # editable install with test/benchmark deps
pytest tests/ -v               # run test suite (CPU, ~2s)
pytest tests/ -v -k gpu        # GPU tests only (need CUDA)
```

## Project structure

```
src/batched_omp/                — pip-installable library (no sklearn dependency in core)
    __init__.py                 — public API: run_omp, omp_v0, omp_v0_blas, BatchedOrthogonalMatchingPursuit
    omp.py                      — run_omp, omp_naive, omp_v0, omp_v0_blas
    sklearn_compat.py           — BatchedOrthogonalMatchingPursuit (drop-in sklearn replacement)
    utils.py                    — batch_mm, innerp, cholesky_solve, elapsed_timer
    blas_kernels/
        _kernels.pyx            — Cython BLAS wrappers (daxpy, dgemv, dppsv, idamax)
tests/
    conftest.py                 — shared fixtures
    test_sklearn_compat.py      — 75 tests: correctness, sklearn contract, pipeline, GPU
benchmarks/
    benchmarks.py               — harness + sklearn/SPAMS/cr-sparse baselines + correctness
    plot_results.py             — 3-panel benchmark figure (benchmark_plot.png)
    plot_sweep.py               — N*B*S heatmaps (speedup vs sklearn or SPAMS)
    plot_ablation.py            — ablation study bar charts
    results/                    — timestamped outputs, JSON data
pyproject.toml                  — package metadata, deps, build config
setup.py                        — Cython extension build
```

## Algorithms

Selectable via `run_omp(X, y, n_nonzero_coefs, alg=...)`:

| `alg` | Function | Description |
|-------|----------|-------------|
| `'v0'` (default) | `omp_v0()` | Inverse Cholesky (Zhu et al. 2020). Fastest on CPU and GPU. Uses `torch.baddbmm`, `torch.bmm`, `torch.gather`. |
| `'naive'` | `omp_naive()` | Batched Cholesky. CPU: packed triangle + BLAS. GPU: `torch.linalg.cholesky`. Not user-facing, removed from benchmarks. |
| `'v0_blas'` | `omp_v0_blas()` | NumPy + Cython BLAS variant of v0. Wins on CPU when n_features is large (e.g. 8064). |

## Key design decisions

- **No sklearn dependency in core library.** sklearn is only used in `sklearn_compat.py` (soft import) and `benchmarks/`.
- **run_omp accepts both tensors and ndarrays.** Ndarrays are auto-converted to tensors.
- **GPU is automatic.** Pass CUDA tensors and everything runs on GPU — no flag needed.
- **Cython BLAS wrappers** call scipy's cython_blas/cython_lapack directly. C-contiguous arrays are passed as Fortran layout to BLAS (so `trans='N'` means `A_C^T @ x`).
- **v0_blas returns early** in run_omp — it bypasses the torch normalization/solution path and returns directly as a dense tensor.
- **torch is a soft dependency.** Package imports without PyTorch; `_require_torch()` raises with install instructions on first use.

## Cython BLAS wrappers (_kernels.pyx)

- `argmax_blast()` — batched `idamax` (absolute-value argmax per row)
- `update_projections_blast()` — batched `daxpy` (`proj[i] += coef[i] * D[i]`)
- `update_D_mybest_blast()` — batched `dgemv` (fused: `D[i] = alpha * D[i] + beta * A[i]^T @ x[i]`)
- `ppsv()` — batched packed Cholesky solve (`dppsv`)
- `project_argmax()` — fused `dgemv` + `idamax` (untested)

## Benchmarking

```bash
pip install spams                                    # C++ baseline

python benchmarks/benchmarks.py all                  # realistic configs
python benchmarks/benchmarks.py sweep                # 108-cell N*B*S grid (~20 min)
python benchmarks/benchmarks.py ablation             # batching/Gram/Cholesky contributions

python benchmarks/plot_results.py                    # 3-panel figure
python benchmarks/plot_sweep.py                      # vs sklearn heatmaps
python benchmarks/plot_sweep.py --baseline=spams     # vs SPAMS heatmaps
python benchmarks/plot_ablation.py                   # ablation bar charts
```

### Baselines

- **sklearn**: `OrthogonalMatchingPursuit` — single-sample loop, CPU only. Reference baseline.
- **SPAMS**: C++ with OpenMP. Fastest CPU implementation. 3 warmup + 3 timed runs.
- **cr-sparse**: JAX GPU. Crashes on overcomplete dictionaries. Not a serious competitor.

### Current results (AWS g7e.8xlarge)

Hardware: Intel Xeon Platinum 8559C, NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM)

| Config | sklearn | SPAMS | Best CPU | GPU | GPU vs SPAMS |
|--------|---------|-------|----------|-----|-------------|
| Image patches | 594 sps | 7,139 | 3,257 (v0) | 183,904 | 25.8x faster |
| Face recognition | 352 sps | 1,717 | 1,731 (BLAS) | 25,158 | 14.6x faster |
| Audio | 207 sps | 2,359 | 926 (v0) | 28,336 | 12.0x faster |

GPU wins 108/108 sweep cells (100%) vs SPAMS. No OOM on any config.
