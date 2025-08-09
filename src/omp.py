import torch
import numpy as np

from _cython.cythomp import argmax_blast, ppsv

def innerp(x, y=None, out=None):
    if y is None:
        y = x
    if out is not None:
        out = out[:, None, None]  # Add space for two singleton dimensions.
    return torch.matmul(x[..., None, :], y[..., :, None], out=out)[..., 0, 0]


def cholesky_solve(ATA, ATy):
    if ATA.dtype == torch.half or ATy.dtype == torch.half:
        return ATy.to(torch.float).cholesky_solve(torch.cholesky(ATA.to(torch.float))).to(ATy.dtype)
    return ATy.cholesky_solve(torch.cholesky(ATA)).to(ATy.dtype)


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

    def batch_mm(matrix, matrix_batch, return_contiguous=True):
        """
        :param matrix: Sparse or dense matrix, size (m, n).
        :param matrix_batch: Batched dense matrices, size (b, n, k).
        :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
        """
        # One dgemm is faster than many dgemv.
        # From https://github.com/pytorch/pytorch/issues/14489#issuecomment-607730242
        batch_size = matrix_batch.shape[0]
        # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
        vectors = matrix_batch.transpose([1, 0, 2]).reshape(matrix.shape[1], -1)

        # A matrix-matrix product is a batched matrix-vector product of the columns.
        # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k).
        if return_contiguous:
            result = np.empty_like(matrix_batch, shape=(batch_size, matrix.shape[0], matrix_batch.shape[2]))
            np.matmul(matrix, vectors, out=result.transpose([1, 0, 2]).reshape(matrix.shape[0], -1))
        else:
            result = (matrix @ vectors).reshape(matrix.shape[0], batch_size, -1).transpose([1, 0, 2])

        return result

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

        # UPDATE ATA based on AT or precomputed XTX.
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
            ATA[:, :k, k] = ATA[:, k, :k]  # Copy lower triangle to upper triangle.
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
                    assert False, "inverse_cholesky=False with tol != None is not handled yet"
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
            if k:  # Could maybe get a speedup from triangular mat mul kernel.
                torch.bmm(D_mybest_maxindices[:, None, :], F[:, :k, :], out=F[:, k, None, :])
                F[:, k, k] = temp_F_k_k[..., 0]
    else: # FIXME: else branch will not execute if n_nonzero_coefs=0, so solutions is undefined.
        # Normal exit, used when tolerance=None.
        if inverse_cholesky:
            solutions = F.permute(0, 2, 1) @ a_F.squeeze(-1).transpose(1, 0)[:, :, None]
        else:
            # Solving the problem in the end without using inverse Cholesky.
            AT = X.T[sets.T]
            solutions = cholesky_solve(AT @ AT.permute(0, 2, 1), AT @ y.T[:, :, None])

    return sets.t(), solutions, None
