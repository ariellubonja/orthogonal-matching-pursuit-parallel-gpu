import torch
import numpy as np
from test_omp import faster_projections, get_max_projections_blas

# Location of 1st gpu device
gpu = "cuda:0"


def omp_naive_gpu(X, y, n_nonzero_coefs):
    print("omp_naive more optimized version running on ", X.device)
    Xt = X.T.contiguous()
    y = y.T.contiguous()
    r = torch.clone(y)  # Maybe no transpose? Remove this line?
    # Int64 is needed here
    sets = torch.zeros((n_nonzero_coefs, r.shape[0]), dtype=torch.int64).to(gpu)
    problems = torch.zeros((r.shape[0], n_nonzero_coefs, X.shape[0]), dtype=torch.float64).to(gpu)
    As = torch.tensor(np.repeat(np.identity(n_nonzero_coefs, dtype=np.float64)[np.newaxis], r.shape[0], 0)).to(gpu)
    # solutions = np.zeros((r.shape[0], n_nonzero_coefs))
    xests = torch.zeros((r.shape[0], X.shape[1]), dtype=torch.float64).to(gpu)
    for k in range(n_nonzero_coefs):
        #t1 = time.time()
        projections = (Xt @ r[:, :, None]).squeeze(-1)  # X.shape[0] * (2*X.shape[1]-1) * r.shape[0] = O(bNM), where X is an MxN matrix, N>M.

        best_idxs = torch.abs(projections).squeeze(-1).argmax(1)
        sets[k, :] = best_idxs
        best = Xt[best_idxs, :]  # A mess...
        problems[:, k, :] = best
        current_problemst = problems[:, :k+1, :]
        # current_problemst.transpose(2,1) I think this is the equivalent Pytorch
        # current_problems = current_problemst.transpose([0, 2, 1])
        current_problems = current_problemst.transpose(2,1)
        # TODO: We have already computed the result of current_problemst @ y[:, :, None]. (It is in projections I believe)
        #       And similarly for the hermitian - it can be constructed from XTX.
        # LAPACK has dgesvx/dgelsy (seems gelsy is newer/better) - gesv is for solving, these other ones work for least squares!
        #  I think linalg.solve calls gesv, which uses LU factorization. https://stackoverflow.com/questions/55367024/fastest-way-of-solving-linear-least-squares
        update = (current_problemst[:, :k, :] @ best[..., None]).squeeze(-1)
        # Our algorithm could be so much faster if the lhs were not (likely) all different.
        #  like in batch_mm, we could then treat the rhs as a single matrix.
        As[:, k, :k] = update  # We only have to update the lower triangle for sposv to work!
        # if True:
        As[:, :k, k] = update
        solutions = torch.solve(
            # As[:, :k+1, :k+1],
            # current_problemst is in float32. Doesn't like that
            # TODO There is an error here when k=1
            current_problemst @ y[:, :, None],
            As[:, :k+1, :k+1])
        solutions = torch.squeeze(solutions.solution,2)  # O(bk^2) memory.
        # else:
        #     # ^ This is faster for small matrices (Python overhead most likely, but may also just be complexity)
        #     solutions = solve_lstsq(current_problems, As[:, :k+1, :k+1], y)
        r = y - torch.squeeze(current_problems @ solutions[:, :, None])

        # maybe memoize in case y is large, such that probability of repeats is significant.
        # If we can move all the batched stuff to RHS (using dense representations) it may be faster.
        # maybe memoize in case y is large, such that probability of repeats is significant. <- In case sparsity is low (e.g. < 32), this may be the fastest method, only limited by memory. (binomial(n, n*0.2))   else:
        # https://en.wikipedia.org/wiki/Singular_value_decomposition#Relation_to_eigenvalue_decomposition
        # https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        # Test if _^2 is faster than abs(_)
    else:
        # np.put_along_axis(xests.T, sets.T, solutions, 1)
        # np.put_along_axis(xests, sets.T, solutions, 1)
        xests.scatter_(1, sets.T, solutions)

    return xests


def omp_naive_blas_gpu_unoptimized(X, y, n_nonzero_coefs):
    print("omp_naive running on ", X.device)
    Xt = X.T.contiguous()
    y = y.T.contiguous()
    r = torch.clone(y)  # Maybe no transpose? Remove this line?
    sets = torch.zeros((n_nonzero_coefs, r.shape[0]), dtype=torch.int64).to(gpu)
    problems = torch.zeros((r.shape[0], n_nonzero_coefs, X.shape[0]), dtype=torch.float64).to(gpu)
    As = torch.tensor(np.repeat(np.identity(n_nonzero_coefs, dtype=np.float64)[np.newaxis], r.shape[0], 0)).to(gpu)
    # solutions = np.zeros((r.shape[0], n_nonzero_coefs))
    xests = torch.zeros((r.shape[0], X.shape[1]), dtype=torch.float64).to(gpu)
    for k in range(n_nonzero_coefs):
        #t1 = time.time()
        projections = faster_projections(Xt, r).squeeze(-1)  # X.shape[0] * (2*X.shape[1]-1) * r.shape[0] = O(bNM), where X is an MxN matrix, N>M.
        #t2 = time.time()
        #print((X.shape[0] * (2*X.shape[1]-1) * r.shape[0] * 1e-9)/(t2-t1), 'GFLOPS')
        # TODO switching between devices. fix this horrible thing
        best_idxs = torch.tensor(get_max_projections_blas(torch.clone(projections).to("cpu").numpy()))  # best_idxs = np.abs(projections).squeeze(-1).argmax(1)  # O(bN), https://scikit-cuda.readthedocs.io/en/latest/generated/skcuda.misc.maxabs.html
        # best_idxs = torch.abs(projections).squeeze(-1).argmax(1)
        sets[k, :] = best_idxs
        best = Xt[best_idxs, :]  # A mess...
        problems[:, k, :] = best
        current_problemst = problems[:, :k+1, :]
        # current_problemst.transpose(2,1) I think this is the equivalent Pytorch
        # current_problems = current_problemst.transpose([0, 2, 1])
        current_problems = current_problemst.transpose(2,1)
        # TODO: We have already computed the result of current_problemst @ y[:, :, None]. (It is in projections I believe)
        #       And similarly for the hermitian - it can be constructed from XTX.
        # LAPACK has dgesvx/dgelsy (seems gelsy is newer/better) - gesv is for solving, these other ones work for least squares!
        #  I think linalg.solve calls gesv, which uses LU factorization. https://stackoverflow.com/questions/55367024/fastest-way-of-solving-linear-least-squares
        update = (current_problemst[:, :k, :] @ best[..., None]).squeeze(-1)
        # Our algorithm could be so much faster if the lhs were not (likely) all different.
        #  like in batch_mm, we could then treat the rhs as a single matrix.
        As[:, k, :k] = update  # We only have to update the lower triangle for sposv to work!
        # if True:
        As[:, :k, k] = update
        solutions = torch.solve(
            # As[:, :k+1, :k+1],
            # current_problemst is in float32. Doesn't like that
            # TODO There is an error here when k=1
            current_problemst @ y[:, :, None],
            As[:, :k+1, :k+1])
        solutions = torch.squeeze(solutions.solution,2)  # O(bk^2) memory.
        # else:
        #     # ^ This is faster for small matrices (Python overhead most likely, but may also just be complexity)
        #     solutions = solve_lstsq(current_problems, As[:, :k+1, :k+1], y)
        r = y - torch.squeeze(current_problems @ solutions[:, :, None])

        # maybe memoize in case y is large, such that probability of repeats is significant.
        # If we can move all the batched stuff to RHS (using dense representations) it may be faster.
        # maybe memoize in case y is large, such that probability of repeats is significant. <- In case sparsity is low (e.g. < 32), this may be the fastest method, only limited by memory. (binomial(n, n*0.2))   else:
        # https://en.wikipedia.org/wiki/Singular_value_decomposition#Relation_to_eigenvalue_decomposition
        # https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        # Test if _^2 is faster than abs(_)
    else:
        # np.put_along_axis(xests.T, sets.T, solutions, 1)
        # np.put_along_axis(xests, sets.T, solutions, 1)
        xests.scatter_(1, sets.T, solutions)

    return xests
