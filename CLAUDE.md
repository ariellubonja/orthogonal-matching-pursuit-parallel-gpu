# Batched Orthogonal Matching Pursuit (CPU + GPU)

Paper by Ariel Lubonja + Sebastian Praesius. Claims up to 200x GPU speedup and 3-8x CPU speedup over sklearn.

## Project structure

```
src/main.py          — core OMP implementations + entry point
src/benchmark.py     — benchmarking harness (realistic + paper configs)
src/plot_results.py  — generates 3-panel benchmark figure
src/test_omp.py      — legacy dev scratchpad (imported for omp_naive)
src/cython/test.pyx  — Cython BLAS wrappers (daxpy, dgemv, dppsv, idamax)
src/cython/setup.py  — Cython build config
results/             — benchmark outputs, plots, paper PDF
```

## OMP algorithms

Three implementations, selectable via `run_omp(..., alg=)`:

| Algorithm | Function | Where | Description |
|-----------|----------|-------|-------------|
| `sklearn` | sklearn's `OrthogonalMatchingPursuit` | main.py:66 | Single-sample, sequential. Baseline. |
| `naive` | `omp_naive()` | main.py:124 | Batched Cholesky. CPU uses packed triangle + BLAS (`ppsv`, `argmax_blast`). GPU uses `torch.linalg.cholesky`. |
| `v0` | `omp_v0()` | main.py:225 | Batched inverse Cholesky. Avoids solving the system each iteration. Uses `torch.gather`, `torch.baddbmm`, `torch.bmm`. Fastest on both CPU and GPU. |
| `v0_blas` | `omp_v0_blas()` | benchmark.py:18 | Experimental numpy + Cython BLAS variant of v0. Uses `update_D_mybest_blast` (dgemv) and `update_projections_blast` (daxpy). Faster than torch v0 when n_features is large (e.g. face_recognition 8064-dim). |

### Key helpers (main.py)

- `run_omp()` (line 31) — unified entry point. Handles normalization, precompute, dispatches to algorithm.
- `batch_mm()` (line 88) — batched matrix multiply via reshape trick for numpy.
- `innerp()` (line 111) — batched inner product.
- `cholesky_solve()` (line 118) — `torch.linalg.cholesky` + `cholesky_solve`. Handles half precision.

### Cython BLAS wrappers (test.pyx)

- `argmax_blast()` (line 103) — batched `idamax` (absolute-value argmax per row)
- `update_projections_blast()` (line 21) — batched `daxpy` (in-place `proj[i] += coef[i] * D[i]`)
- `update_D_mybest_blast()` (line 40) — batched `dgemv` (fused scale + mat-vec: `D[i] = alpha * D[i] + beta * A[i]^T @ x[i]`)
- `ppsv()` (line 83) — batched packed Cholesky solve (`dppsv`)
- `project_argmax()` (line 121) — fused `dgemv` + `idamax` (untested in production)

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
4. **Benchmarking** — `benchmark.py` with 3 realistic configs + paper Fig 1 reproduction. Results written to timestamped files in `results/`.
5. **Cython BLAS bug fixes** — fixed `ldaA=2048` hardcode, `strides[1]` OOB on 1D memoryviews, wrong `trans='T'` (should be `'N'` for C-contiguous → Fortran layout).
6. **Removed dead code** — deleted `1d_omp_gpu_v12/` (CUDA 4.0 legacy, nothing to port), removed unnecessary ATA upper-triangle copy in GPU naive path.

## Pending

- **Plan sklearn PR** — contribution of batched OMP to scikit-learn
- **Stronger baselines** — consider SPAMS (Mairal et al.) for workshop paper

## How to run

```bash
# Benchmarks (disable CPU features first for stable results)
sudo ~/prog/set_cpu_e_features.sh --disable
.venv-py312/bin/python3 src/benchmark.py all        # all configs + GPU
.venv-py312/bin/python3 src/benchmark.py paper --no-gpu
sudo ~/prog/set_cpu_e_features.sh --enable

# Rebuild Cython after editing test.pyx
.venv-py312/bin/python3 -c "from Cython.Build import cythonize; cythonize('src/cython/test.pyx', force=True, language_level='3')"
.venv-py312/bin/python3 src/cython/setup.py build_ext --inplace

# Plot results
.venv-py312/bin/python3 src/plot_results.py
```

## Branches

- `main` — py312 port (active development)
- `py39` — original py39 code (archived)
