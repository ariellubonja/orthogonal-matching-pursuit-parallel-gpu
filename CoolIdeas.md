# Cool Ideas from test_omp.py

Ideas extracted from comments and experimental code in `src/test_omp.py`.

## Worth trying

### SOMP (Simultaneous Orthogonal Matching Pursuit)
> Line 58: "all the times we insert [..., None] should just be replaced by the input already having a singleton dimension — then after projection we do a sum of absolutes [asum] before doing idamax. Everything else will work as-is."

The code is 90% SOMP-ready. Multiple measurement vectors share the same support, and the only change is summing absolute projections across channels before argmax. Natural extension for the paper. Ref: https://arxiv.org/pdf/1506.05324.pdf

### Cache `1/temp_F_k_k` not just `1/sqrt`
> Lines 318, 362: "It may be faster to save or use 1/* and not just 1/sqrt(*) — since many places this is multiplied twice!"

`temp_F_k_k = rsqrt(...)` is multiplied into D_mybest, projections, and F. Caching the reciprocal squared could save a redundant multiply per iteration. Micro-optimization but free.

### `dsyrk` for Gram matrix (A^T @ A)
> Line 490: "Use dsyrk to calculate matrix times own transpose. Then solve with posv/gelsy."

BLAS `dsyrk` computes symmetric rank-k updates in half the FLOPs of a full matmul. Currently `X.T @ X` uses general `dgemm`. Relevant for the naive path's Gram matrix computation.

### `get_lapack_funcs` for runtime BLAS dispatch
> Line 66: "Use get_lapack_funcs to select appropriate and fastest functions (and possibly take into account work-memory)"

scipy can auto-select the best LAPACK routine for the dtype/hardware. Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.get_lapack_funcs.html

### Multiprocessing for large CPU batches
> Lines 820-828: skeleton using `multiprocessing.Pool` + `np.array_split(y, no_workers)`.

Split the batch across CPU workers. Never finished. Could help for very large batches where single-core BLAS is the bottleneck. Note (line 822): "Gramian can be calculated once locally, and sent to each thread."

## Already done

- **Packed Cholesky (`ppsv`)** — in main.py's naive CPU path (line 213)
- **`batch_mm` reshape trick** — in main.py (line 88), avoids many dgemv with one dgemm
- **numpy-faster-than-torch on CPU** (lines 646, 667) — exploited in naive CPU path: numpy matmul for ATA update, numpy subtract for residual
- **`argmax_blast` via BLAS `idamax`** — in main.py naive CPU path (line 180) and benchmark.py v0_blas
- **`update_projections_blast` / `update_D_mybest_blast`** — Cython daxpy/dgemv wrappers, used in benchmark.py's v0_blas. Fixed bugs: hardcoded ldaA=2048, OOB strides[1], wrong trans='T'
- **Cholesky vs LU solve** (line 438) — tested, cholesky_solve ~1% faster than torch.solve. Using cholesky in main.py.
- **Memoization** (lines 48-53) — explicitly concluded "not going to give any significant speedups" for random dictionaries. Probability of repeated support paths is ~1/N per iteration.

## Out of scope / theoretical

### Column normalization and pseudoinverse interaction
> Lines 60-63: "It seems since we normalize columns, we should have to take this into account somewhere? Why do we normalize columns again?"

Theoretical concern: if OMP operates on column-normalized X, does the final pseudoinverse need correction? Currently `run_omp` un-normalizes solutions at line 75 (`solutions /= normalize[sets]`), which should handle this. Worth a sanity check but likely fine.

### MP vs OMP
> Line 56: reference to Matching Pursuit (no orthogonalization). Could be useful as a cheaper baseline for comparison but not the focus.

### `gesv` vs `posv` for small systems
> Lines 110-111: "If it is large posv should always be faster. But if it is small, maybe gesv is better?"

For the small k×k systems in OMP inner loops, the overhead difference between general solve and Cholesky solve is negligible.
