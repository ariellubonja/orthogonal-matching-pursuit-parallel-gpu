# Batched Orthogonal Matching Pursuit (CPU + GPU)

Paper by Ariel Lubonja + Sebastian Praesius. Claims up to 200x GPU speedup and 3-8x CPU speedup over sklearn.

## Project structure

```
pyproject.toml                        — package metadata + build deps
setup.py                              — Cython extension build
src/batched_omp/
    __init__.py                       — exports run_omp, omp_naive, omp_v0, utils
    omp.py                            — core OMP implementations (naive, v0)
    utils.py                          — batch_mm, innerp, cholesky_solve, elapsed_timer
    blas_kernels/
        __init__.py                   — re-exports from _kernels
        _kernels.pyx                  — Cython BLAS wrappers (daxpy, dgemv, dppsv, idamax)
benchmarks/
    benchmark.py                      — benchmarking harness (realistic + paper configs)
    plot_results.py                   — generates 3-panel benchmark figure
    results/                          — benchmark outputs + plots
CoolIdeas.md                          — future optimization ideas
```

## OMP algorithms

Two implementations in the library, selectable via `run_omp(..., alg=)`:

| Algorithm | Function | Where | Description |
|-----------|----------|-------|-------------|
| `naive` | `omp_naive()` | omp.py | Batched Cholesky. CPU uses packed triangle + BLAS (`ppsv`, `argmax_blast`). GPU uses `torch.linalg.cholesky`. |
| `v0` | `omp_v0()` | omp.py | Batched inverse Cholesky. Avoids solving the system each iteration. Uses `torch.gather`, `torch.baddbmm`, `torch.bmm`. Fastest on both CPU and GPU. |

Benchmarks-only (not part of library):

| Algorithm | Function | Where | Description |
|-----------|----------|-------|-------------|
| `sklearn` | `run_sklearn()` | benchmarks/benchmark.py | sklearn's `OrthogonalMatchingPursuit`. Baseline. |
| `v0_blas` | `omp_v0_blas()` | benchmarks/benchmark.py | Experimental numpy + Cython BLAS variant of v0. Faster than torch v0 when n_features is large. |

### Key helpers (utils.py)

- `run_omp()` — unified entry point. Handles normalization, precompute, dispatches to algorithm.
- `batch_mm()` — batched matrix multiply via reshape trick for numpy.
- `innerp()` — batched inner product.
- `cholesky_solve()` — `torch.linalg.cholesky` + `cholesky_solve`. Handles half precision.
- `elapsed_timer()` — context manager for timing.

### Cython BLAS wrappers (_kernels.pyx)

- `argmax_blast()` — batched `idamax` (absolute-value argmax per row)
- `update_projections_blast()` — batched `daxpy` (in-place `proj[i] += coef[i] * D[i]`)
- `update_D_mybest_blast()` — batched `dgemv` (fused scale + mat-vec: `D[i] = alpha * D[i] + beta * A[i]^T @ x[i]`)
- `ppsv()` — batched packed Cholesky solve (`dppsv`)
- `project_argmax()` — fused `dgemv` + `idamax` (untested in production)

## Benchmark results

Hardware: Intel Core Ultra 9 185H (turbo/HT/E-cores disabled), RTX 4060 Laptop (7.62 GiB)
Settings: `tol=None, normalize=False, fit_intercept=False, precompute=True`

### Realistic configs (speedup vs sklearn)

| Config | Dims (feat x comp) | nnz | B | sklearn | naive CPU | v0 CPU | v0 GPU |
|--------|---------------------|-----|------|---------|-----------|--------|--------|
| Image patches | 256 x 1024 | 32 | 5000 | 278 sps | 4.3x | 6.7x | **67.6x** |
| Face recognition | 8064 x 1207 | 30 | 1207 | 271 sps | 0.3x | 1.8x | **15.4x** |
| Audio | 512 x 2048 | 64 | 5000 | 115 sps | 2.0x | 4.0x | **36.6x** |

### Paper Fig 1 (N=8M, S=M/4, B=100)

| M | sklearn | v0 CPU | v0 GPU speedup |
|------|---------|--------|----------------|
| 128 | 0.208s | 0.105s | 14x |
| 256 | 0.622s | 0.265s | 19x |
| 512 | 3.627s | 1.328s | 17x |
| 1024 | 40.3s | 7.131s | 24x |
| 2048 | 295.4s | 44.97s | OOM (6.25 GiB needed) |

### BLAS v0 vs torch v0 (CPU only)

| Config | torch v0 | BLAS v0 | Winner |
|--------|----------|---------|--------|
| Image patches | 2475 sps | 1980 sps | torch (+25%) |
| Face recognition | 783 sps | 1514 sps | **BLAS (+93%)** |
| Audio | 627 sps | 389 sps | torch (+61%) |

BLAS v0 wins when n_features is large (avoids temporaries in dgemv vs torch.baddbmm).

## Completed work

1. **py312 migration** — ported from py39. Fixed `torch.cholesky` -> `torch.linalg.cholesky`, sklearn `normalize` kwarg removal.
2. **Correctness verification** — orthogonality violations ~1e-15, naive/v0 support sets agree on all samples, GPU/CPU agree.
3. **Error measurements** — orthogonality check (`X[:, support]^T @ residual`), NNZ/tol invariant, naive-vs-v0 support diffs.
4. **Benchmarking** — `benchmark.py` with 3 realistic configs + paper Fig 1 reproduction. Results written to timestamped files.
5. **Cython BLAS bug fixes** — fixed `ldaA=2048` hardcode, `strides[1]` OOB on 1D memoryviews, wrong `trans='T'` (should be `'N'` for C-contiguous → Fortran layout).
6. **Removed dead code** — deleted `1d_omp_gpu_v12/`, `test_omp.py`, unnecessary ATA upper-triangle copy in GPU naive path.
7. **Project restructure** — from flat `src/` to pip-installable `src/batched_omp/` package + `benchmarks/`. Library has no sklearn dependency.

## Pending

- **Publish GPU PyPI package**
- **Stronger baselines** — consider SPAMS (Mairal et al.) for workshop paper

## How to run

```bash
# Install (editable, with dev deps for benchmarks)
uv pip install -e ".[dev]" --python .venv-py312/bin/python3

# Benchmarks (disable CPU features first for stable results)
sudo ~/prog/set_cpu_e_features.sh --disable
.venv-py312/bin/python3 benchmarks/benchmark.py all        # all configs + GPU
.venv-py312/bin/python3 benchmarks/benchmark.py paper --no-gpu
sudo ~/prog/set_cpu_e_features.sh --enable

# Rebuild Cython after editing _kernels.pyx
uv pip install -e ".[dev]" --python .venv-py312/bin/python3

# Plot results
.venv-py312/bin/python3 benchmarks/plot_results.py
```

## Branches

- `main` — py312 port (active development)
- `py39` — original py39 code (archived)
