# Batched OMP — Orientation Guide

Paper: https://arxiv.org/abs/2407.06434
By Ariel Lubonja, Sebastian Kazmarek Praesius, and Trac Duy Tran.

## Project structure

```
src/batched_omp/                — pip-installable library (no sklearn dependency)
    omp.py                      — run_omp, omp_naive, omp_v0, omp_v0_blas
    utils.py                    — batch_mm, innerp, cholesky_solve, elapsed_timer
    blas_kernels/
        _kernels.pyx            — Cython BLAS wrappers (daxpy, dgemv, dppsv, idamax)
benchmarks/
    benchmarks.py               — harness + sklearn baseline + correctness checks
    plot_results.py             — 3-panel benchmark figure
    results/                    — timestamped outputs + plots
pyproject.toml                  — package metadata, deps, build config
setup.py                        — Cython extension build
CoolIdeas.md                    — future optimization ideas extracted from old code
```

## Algorithms

Selectable via `run_omp(X, y, n_nonzero_coefs, alg=...)`:

| `alg` | Function | Description |
|-------|----------|-------------|
| `'v0'` (default) | `omp_v0()` | Inverse Cholesky. Fastest on CPU and GPU. Uses `torch.baddbmm`, `torch.bmm`, `torch.gather`. |
| `'naive'` | `omp_naive()` | Batched Cholesky. CPU: packed triangle + BLAS (`ppsv`, `argmax_blast`). GPU: `torch.linalg.cholesky`. |
| `'v0_blas'` | `omp_v0_blas()` | NumPy + Cython BLAS variant of v0. Wins on CPU when n_features is large (e.g. 8064). |

## Key design decisions

- **No sklearn dependency in library.** sklearn is only used in benchmarks/ as a baseline.
- **run_omp accepts both tensors and ndarrays.** Ndarrays are auto-converted to tensors.
- **GPU is automatic.** Pass CUDA tensors and everything runs on GPU — no flag needed.
- **Cython BLAS wrappers** call scipy's cython_blas/cython_lapack directly. C-contiguous arrays are passed as Fortran layout to BLAS (so `trans='N'` means `A_C^T @ x`).
- **v0_blas returns early** in run_omp — it bypasses the torch normalization/solution path and returns directly as a dense tensor.

## Cython BLAS wrappers (_kernels.pyx)

- `argmax_blast()` — batched `idamax` (absolute-value argmax per row)
- `update_projections_blast()` — batched `daxpy` (`proj[i] += coef[i] * D[i]`)
- `update_D_mybest_blast()` — batched `dgemv` (fused: `D[i] = alpha * D[i] + beta * A[i]^T @ x[i]`)
- `ppsv()` — batched packed Cholesky solve (`dppsv`)
- `project_argmax()` — fused `dgemv` + `idamax` (untested)

## How to build and run

```bash
# Install (editable, with dev deps for benchmarks)
uv pip install -e ".[dev]" --python .venv-py312/bin/python3

# Benchmarks (disable CPU features first for stable results)
sudo ~/prog/set_cpu_e_features.sh --disable
.venv-py312/bin/python3 benchmarks/benchmarks.py all
.venv-py312/bin/python3 benchmarks/benchmarks.py image_patches --no-gpu
sudo ~/prog/set_cpu_e_features.sh --enable

# Rebuild Cython after editing _kernels.pyx
uv pip install -e ".[dev]" --python .venv-py312/bin/python3

# Plot results
.venv-py312/bin/python3 benchmarks/plot_results.py
```

## Branches

- `main` — py312, active development
- `py39` — archived original code
