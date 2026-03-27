import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .utils import batch_mm, innerp, cholesky_solve
from .blas_kernels import argmax_blast, ppsv, update_projections_blast, update_D_mybest_blast


def _require_torch():
    if torch is None:
        raise ImportError(
            "PyTorch is required but not installed. "
            "Install it with: pip install batched-omp[gpu]  "
            "or: pip install torch"
        )


def run_omp(X, y, n_nonzero_coefs, precompute=True, tol=0.0, normalize=True, fit_intercept=True, alg='v0'):
    _require_torch()
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X)
        y = torch.as_tensor(y)

    # TODO: Convert to pytorch tensor at this level?
    # We can either return sets, (sets, solutions), or xests
    # These are all equivalent, but are simply more and more dense representations.
    # Given sets and X and y one can (re-)construct xests. The second is just a sparse vector repr.

    # https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/linear_model/_omp.py#L690
    if fit_intercept or normalize:
        X = X.clone()
        assert not isinstance(precompute, torch.Tensor), "If user pre-computes XTX they can also pre-normalize X" \
                                                         " as well, so normalize and fit_intercept must be set false."

    if fit_intercept:
        X = X - X.mean(0)
        y = y - y.mean(1)[:, None]

    # To keep a good condition number on X, especially with Cholesky compared to LU factorization,
    # we should probably always normalize it (OMP is invariant anyways)
    if normalize:
        normalize = (X * X).sum(0).sqrt()  # User can also just optionally supply pre-computed norms.
        X /= normalize[None, :]  # Save compute if already normalized!

    if precompute or alg == 'v0':
        precompute = X.T @ X

    # If n_nonzero_coefs is equal to M, one should just return lstsq
    if alg == 'naive':
        sets, solutions, lengths = omp_naive(X, y, n_nonzero_coefs=n_nonzero_coefs, XTX=precompute, tol=tol)
    elif alg == 'v0':
        sets, solutions, lengths = omp_v0(X, y, n_nonzero_coefs=n_nonzero_coefs, XTX=precompute, tol=tol)
    elif alg == 'v0_blas':
        sets_np, solutions_np = omp_v0_blas(X.numpy() if isinstance(X, torch.Tensor) else X,
                                            y.numpy() if isinstance(y, torch.Tensor) else y,
                                            n_nonzero_coefs=n_nonzero_coefs)
        # omp_v0_blas returns (sets, solutions) as numpy; convert to dense xests
        xests = np.zeros((y.shape[0], X.shape[1]))
        for i in range(y.shape[0]):
            xests[i, sets_np[i]] = solutions_np[i, :, 0]
        return torch.as_tensor(xests)
    else:
        raise ValueError(f"Unknown algorithm: {alg!r}. Use 'naive', 'v0', or 'v0_blas'.")

    solutions = solutions.squeeze(-1)
    if normalize is not False:
        solutions /= normalize[sets]

    xests = y.new_zeros(y.shape[0], X.shape[1])
    if lengths is None:
        xests[torch.arange(y.shape[0], dtype=sets.dtype, device=sets.device)[:, None], sets] = solutions
    else:
        for i in range(y.shape[0]):
            xests[i, sets[i, :lengths[i]]] = solutions[i, :lengths[i]]

    return xests


def omp_naive(X, y, n_nonzero_coefs, tol=None, XTX=None):
    on_cpu = not (y.is_cuda or y.dtype == torch.half)
    # Given X as an MxN array and y as an BxN array, do omp to approximately solve Xb=y.

    # Base variables
    XT = X.contiguous().t()  # Store XT in fortran-order.
    y = y.contiguous()
    r = y.clone()

    sets = y.new_zeros((n_nonzero_coefs, y.shape[0]), dtype=torch.long).t()
    if tol:
        result_sets = sets.new_zeros(y.shape[0], n_nonzero_coefs)
        result_lengths = sets.new_zeros(y.shape[0])
        result_solutions = y.new_zeros((y.shape[0], n_nonzero_coefs, 1))
        original_indices = torch.arange(y.shape[0], dtype=sets.dtype, device=sets.device)

    # Trade b*k^2+bk+bkM = O(bkM) memory for much less compute time. (This has to be done anyways since we are batching,
    # otherwise one could just permute columns of X in-place as in https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/linear_model/_omp.py#L28 )
    ATs = y.new_zeros(r.shape[0], n_nonzero_coefs, X.shape[0])
    ATys = y.new_zeros(r.shape[0], n_nonzero_coefs, 1)
    ATAs = torch.eye(n_nonzero_coefs, dtype=y.dtype, device=y.device)[None].repeat(r.shape[0], 1, 1)
    if on_cpu:
        # For CPU it is faster to use a packed representation of the lower triangle in ATA.
        tri_idx = torch.tril_indices(n_nonzero_coefs, n_nonzero_coefs, device=sets.device, dtype=sets.dtype)
        ATAs = ATAs[:, tri_idx[0], tri_idx[1]]

    solutions = y.new_zeros((r.shape[0], 0))

    for k in range(n_nonzero_coefs+bool(tol)):
        # STOPPING CRITERIA
        if tol:
            problems_done = innerp(r) <= tol
            if k == n_nonzero_coefs:
                problems_done[:] = True

            if problems_done.any():
                remaining = ~problems_done

                orig_idxs = original_indices[problems_done]
                result_sets[orig_idxs, :k] = sets[problems_done, :k]
                result_solutions[orig_idxs, :k] = solutions[problems_done]
                result_lengths[orig_idxs] = k
                original_indices = original_indices[remaining]

                # original_indices = original_indices[remaining]
                ATs = ATs[remaining]
                ATys = ATys[remaining]
                ATAs = ATAs[remaining]
                sets = sets[remaining]
                y = y[remaining]
                r = r[remaining]
                if problems_done.all():
                    return result_sets, result_solutions, result_lengths
        # GET PROJECTIONS AND INDICES TO ADD
        if on_cpu:
            projections = batch_mm(XT.numpy(), r[:, :, None].numpy())
            argmax_blast(projections.squeeze(-1), sets[:, k].numpy())
        else:
            projections = XT @ r[:, :, None]
            sets[:, k] = projections.abs().sum(-1).argmax(-1)  # Sum is just a squeeze, but would be relevant in SOMP.

        # UPDATE AT
        AT = ATs[:, :k + 1, :]
        updateA = XT[sets[:, k], :]
        AT[:, k, :] = updateA

        # UPDATE ATy based on AT
        ATy = ATys[:, :k + 1]
        innerp(updateA, y, out=ATy[:, k, 0])

        # UPDATE ATA based on AT or XTX.
        if on_cpu:
            packed_idx = k * (k - 1) // 2
            if XTX is not None:  # Update based on precomputed XTX.
                ATAs.t()[k + packed_idx:packed_idx + 2 * k + 1, :].t().numpy()[:] = XTX[sets[:, k, None], sets[:, :k + 1]]
            else:
                np.matmul(AT[:, :k + 1, :].numpy(), updateA[:, :, None].numpy(),
                          out=ATAs.t()[k + packed_idx:packed_idx + 2 * k + 1, :].t()[:, :, None].numpy())
        else:
            ATA = ATAs[:, :k + 1, :k + 1]
            if XTX is not None:
                ATA[:, k, :k + 1] = XTX[sets[:, k, None], sets[:, :k + 1]]
            else:
                # Update ATAs by adding the new column of inner products.
                torch.bmm(AT[:, :k + 1, :], updateA[:, :, None], out=ATA[:, k, :k + 1, None])

        # SOLVE ATAx = ATy.
        if on_cpu:
            solutions = ATy.permute(0, 2, 1).clone().permute(0, 2, 1)  # Get a copy.
            ppsv(ATAs.t()[:packed_idx + 2 * k + 1, :].t().contiguous().numpy(), solutions.numpy())
        else:
            solutions = cholesky_solve(ATA, ATy)

        # FINALLY, GET NEW RESIDUAL r=y-Ax
        if on_cpu:
            np.subtract(y.numpy(), (AT.permute(0, 2, 1).numpy() @ solutions.numpy()).squeeze(-1), out=r.numpy())
        else:
            torch.baddbmm(y[:, :, None], AT.permute(0, 2, 1), solutions, beta=-1, out=r[:, :, None])

    return sets, solutions, None


def omp_v0(X, y, XTX, n_nonzero_coefs=None, tol=None, inverse_cholesky=True):
    B = y.shape[0]
    normr2 = innerp(y)  # Norm squared of residual.
    projections = (X.transpose(1, 0) @ y[:, :, None]).squeeze(-1)
    sets = y.new_zeros(n_nonzero_coefs, B, dtype=torch.int64)

    if inverse_cholesky:
        # Doing the inverse-cholesky iteratively uses more memory,
        # but takes less time than waiting till solving the problem in the end it seems.
        # (Of course this may also just be because we have not optimized it extensively,
        #  but also since F is triangular it could be faster to multiply, prob. not on GPU tho.)
        F = torch.eye(n_nonzero_coefs, dtype=y.dtype, device=y.device).repeat(B, 1, 1)
        a_F = y.new_zeros(n_nonzero_coefs, B, 1)

    D_mybest = y.new_empty(B, n_nonzero_coefs, XTX.shape[0])
    temp_F_k_k = y.new_ones((B, 1))

    if tol:
        result_lengths = sets.new_zeros(y.shape[0])
        result_solutions = y.new_zeros((y.shape[0], n_nonzero_coefs, 1))
        finished_problems = sets.new_zeros(y.shape[0], dtype=torch.bool)

    for k in range(n_nonzero_coefs+bool(tol)):
        # STOPPING CRITERIA
        if tol:
            problems_done = normr2 <= tol
            if k == n_nonzero_coefs:
                problems_done[:] = True

            if problems_done.any():
                new_problems_done = problems_done & ~finished_problems
                finished_problems.logical_or_(problems_done)
                result_lengths[new_problems_done] = k
                if inverse_cholesky:
                    result_solutions[new_problems_done, :k] = F[new_problems_done, :k, :k].permute(0, 2, 1) @ a_F[:k, new_problems_done].permute(1, 0, 2)
                else:
                    assert False, "inverse_cholesky=False with tol != None is not handled"
                if problems_done.all():
                    return sets.t(), result_solutions, result_lengths

        sets[k] = projections.abs().argmax(1)
        # D_mybest[:, k, :] = XTX[gamma[k], :]  # Same line as below, but significantly slower. (prob. due to the intermediate array creation)
        torch.gather(XTX, 0, sets[k, :, None].expand(-1, XTX.shape[1]), out=D_mybest[:, k, :])
        if k:
            D_mybest_maxindices = D_mybest.permute(0, 2, 1)[torch.arange(D_mybest.shape[0], dtype=sets.dtype, device=sets.device), sets[k], :k]
            torch.rsqrt(1 - innerp(D_mybest_maxindices), out=temp_F_k_k[:, 0])  # torch.exp(-1/2 * torch.log1p(-inp), temp_F_k_k[:, 0])
            D_mybest_maxindices *= -temp_F_k_k  # minimal operations, exploit linearity
            D_mybest[:, k, :] *= temp_F_k_k
            D_mybest[:, k, :, None].baddbmm_(D_mybest[:, :k, :].permute(0, 2, 1), D_mybest_maxindices[:, :, None])


        temp_a_F = temp_F_k_k * torch.gather(projections, 1, sets[k, :, None])
        normr2 -= (temp_a_F * temp_a_F).squeeze(-1)
        projections -= temp_a_F * D_mybest[:, k, :]
        if inverse_cholesky:
            a_F[k] = temp_a_F
            if k:  # Could maybe a speedup from triangular mat mul kernel.
                torch.bmm(D_mybest_maxindices[:, None, :], F[:, :k, :], out=F[:, k, None, :])
                F[:, k, k] = temp_F_k_k[..., 0]
    else:
        if inverse_cholesky:
            solutions = F.permute(0, 2, 1) @ a_F.squeeze(-1).transpose(1, 0)[:, :, None]
        else:
            AT = X.T[sets.T]
            solutions = cholesky_solve(AT @ AT.permute(0, 2, 1), AT @ y[:, :, None])

    return sets.t(), solutions, None


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
