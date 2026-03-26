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
    benchmarks.py               — harness + sklearn/SPAMS/cr-sparse baselines + correctness
    plot_results.py             — 3-panel benchmark figure (benchmark_plot.png)
    plot_sweep.py               — N×B×S heatmaps (speedup vs sklearn or SPAMS)
    plot_ablation.py            — ablation study bar charts
    results/                    — timestamped outputs, JSON data
    results/heatmaps/           — generated plot PNGs
    results/cpu_e_features_disabled/ — archived laptop results with E-cores/turbo/HT disabled
pyproject.toml                  — package metadata, deps, build config
setup.py                        — Cython extension build
CoolIdeas.md                    — future optimization ideas extracted from old code
```

## Algorithms

Selectable via `run_omp(X, y, n_nonzero_coefs, alg=...)`:

| `alg` | Function | Description |
|-------|----------|-------------|
| `'v0'` (default) | `omp_v0()` | Inverse Cholesky (Zhu et al. 2020). Fastest on CPU and GPU. Uses `torch.baddbmm`, `torch.bmm`, `torch.gather`. |
| `'naive'` | `omp_naive()` | Batched Cholesky. CPU: packed triangle + BLAS (`ppsv`, `argmax_blast`). GPU: `torch.linalg.cholesky`. Not user-facing, removed from benchmarks. |
| `'v0_blas'` | `omp_v0_blas()` | NumPy + Cython BLAS variant of v0. Wins on CPU when n_features is large (e.g. 8064). |

## Key design decisions

- **No sklearn dependency in library.** sklearn is only used in benchmarks/ as a baseline.
- **run_omp accepts both tensors and ndarrays.** Ndarrays are auto-converted to tensors.
- **GPU is automatic.** Pass CUDA tensors and everything runs on GPU — no flag needed.
- **Cython BLAS wrappers** call scipy's cython_blas/cython_lapack directly. C-contiguous arrays are passed as Fortran layout to BLAS (so `trans='N'` means `A_C^T @ x`).
- **v0_blas returns early** in run_omp — it bypasses the torch normalization/solution path and returns directly as a dense tensor.

## Benchmarking

### Benchmark commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"
# Also install SPAMS: pip install spams

# Run realistic configs (image patches, face recognition, audio)
python benchmarks/benchmarks.py all

# Parameter sweep: 108 cells, N×B×S grid (~20 min)
python benchmarks/benchmarks.py sweep
python benchmarks/plot_sweep.py              # vs sklearn heatmaps
python benchmarks/plot_sweep.py --baseline=spams  # vs SPAMS heatmaps

# Ablation study: batching, Gram precomputation, inverse Cholesky
python benchmarks/benchmarks.py ablation
python benchmarks/plot_ablation.py

# Generate 3-panel benchmark figure
python benchmarks/plot_results.py
```

### What benchmarks run

- **Realistic configs**: image_patches (256×1024), face_recognition (8064×1207), audio (512×2048)
- **Sweep**: N=[64..4096], B=[10..5000], S=[8,32,64], M=N/4. Algorithms: sklearn, SPAMS, v0 CPU, v0 BLAS, v0 GPU. GPU runs 3 times for variance estimation.
- **Ablation**: 4 configs × 3 axes (batching, Gram precompute, inverse Cholesky) on CPU and GPU.
- **Naive algorithm was removed from benchmarks** — it's strictly slower than v0 on all configs.

### Baselines

- **sklearn**: OrthogonalMatchingPursuit — single-sample loop, CPU only. Our reference baseline.
- **SPAMS**: C++ with OpenMP (`pip install spams`). Fastest CPU implementation. 3 warmup + 3 timed runs.
- **cr-sparse**: JAX GPU. Crashes on overcomplete dictionaries (our sweep is 4x overcomplete). Has numerical issues (orthogonality violations up to 3.6 vs our 1e-15). Not a serious competitor.

### Current results (laptop, CPU E-features ENABLED = real-world config)

Hardware: Intel Core Ultra 9 185H (6P+8E cores), NVIDIA RTX 4060 Laptop (8 GB VRAM)

| Config | sklearn | SPAMS | Best CPU | GPU | GPU vs SPAMS |
|--------|---------|-------|----------|-----|-------------|
| Image patches | 533 sps | 23,106 | 2,227 (v0) | 22,633 | tied |
| Face recognition | 482 sps | 1,274 | 1,470 (BLAS) | 4,285 | **3.4x faster** |
| Audio | 103 sps | 4,490 | 504 (v0) | OOM | — |

Sweep: GPU wins 67/108 cells (62%) vs SPAMS. Wins 89% when N >= 2048.

## Next task: AWS GPU benchmarks

### Why

The RTX 4060 Laptop (8 GB VRAM) OOMs on:
- Audio config (512×2048, S=64, B=5000)
- 3 large sweep cells (N=4096, B=5000)

A GPU with more VRAM would fix all OOM cases. Based on sweep patterns (GPU wins 2-9x at large N), these configs should show GPU beating SPAMS. A faster GPU also widens the margin on all existing configs.

### Recommended instance

**g7e.8xlarge**. Fixes all OOM, representative of what practitioners use.

### What to run on AWS

```bash
# 1. Clone repo and install
git clone <repo-url> && cd orthogonal-matching-pursuit-parallel-gpu
pip install -e ".[dev]"
pip install spams

# 2. Run realistic configs
python benchmarks/benchmarks.py all

# 3. Run sweep (generates JSON for heatmaps)
python benchmarks/benchmarks.py sweep

# 4. Run ablation
python benchmarks/benchmarks.py ablation

# 5. Generate all plots
python benchmarks/plot_results.py
python benchmarks/plot_sweep.py
python benchmarks/plot_sweep.py --baseline=spams
python benchmarks/plot_ablation.py
```

### What to update after AWS run

1. Update `benchmarks/plot_results.py` — the `realistic` data array (line ~25) with new sps numbers
2. Update `README.md` — speedup table, tagline numbers
3. Keep laptop results in `results/cpu_e_features_disabled/` for comparison

## Cython BLAS wrappers (_kernels.pyx)

- `argmax_blast()` — batched `idamax` (absolute-value argmax per row)
- `update_projections_blast()` — batched `daxpy` (`proj[i] += coef[i] * D[i]`)
- `update_D_mybest_blast()` — batched `dgemv` (fused: `D[i] = alpha * D[i] + beta * A[i]^T @ x[i]`)
- `ppsv()` — batched packed Cholesky solve (`dppsv`)
- `project_argmax()` — fused `dgemv` + `idamax` (untested)

## Branches

- `main` — py312, active development
- `py39` — archived original code
