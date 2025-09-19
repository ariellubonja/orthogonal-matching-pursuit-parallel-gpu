import torch
from sklearn.datasets import make_sparse_coded_signal
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from contextlib import contextmanager
from timeit import default_timer
import argparse

from omp import omp_naive, omp_v0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=100)
    parser.add_argument("--n_features", type=int, default=100)
    parser.add_argument("--n_nonzero_coefs", type=int, default=17)
    parser.add_argument("--n_samples", type=int, default=50)

    return parser.parse_args()


@contextmanager
def elapsed_timer():
    # https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


"""This is a helper function for allowing us to profile both GPU runtime as well as the time it takes to transfer to GPU memory"""
def gpu_transfer_and_alg(X,y, alg, n_nonzero_coefs):
    X_gpu = torch.as_tensor(X, device='cuda', dtype=torch.float)
    y_gpu = torch.as_tensor(y, device='cuda', dtype=torch.float)
    results = run_omp(X_gpu, y_gpu, n_nonzero_coefs, alg=alg)
    results.cpu()
    return results


def run_omp(X, y, n_nonzero_coefs, precompute=True, tol=0.0, normalize=False, fit_intercept=False, alg='naive'):
    """
    Wrapper function to facilitate running benchmarks against our novel OMP implementations
    """
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
    if normalize is True:
        normalize = (X * X).sum(0).sqrt()  # User can also just optionally supply pre-computed norms.
        X /= normalize[None, :]  # Save compute if already normalized!

    if precompute is True or alg == 'v0':
        precompute = X.T @ X

    # If n_nonzero_coefs is equal to M, one should just return lstsq
    if alg == 'naive':
        sets, solutions, lengths = omp_naive(X, y, n_nonzero_coefs=n_nonzero_coefs, XTX=precompute, tol=tol)
    elif alg == 'v0':
        sets, solutions, lengths = omp_v0(X, y, n_nonzero_coefs=n_nonzero_coefs, XTX=precompute, tol=tol)

    solutions = solutions.squeeze(-1)
    if normalize is not False:
        solutions /= normalize[sets]

    xests = y.new_zeros(y.shape[0], X.shape[1])
    if lengths is None:
        xests[torch.arange(y.shape[0], dtype=sets.dtype, device=sets.device)[:, None], sets] = solutions
    else:
        for i in range(y.shape[0]):
            # print(sets.shape, xests[i, sets[i, :lengths[i]]].shape)
            # xests[i].scatter_(-1, sets[i, :lengths[i]], solutions[i, :lengths[i]])
            xests[i, sets[i, :lengths[i]]] = solutions[i, :lengths[i]]

    return xests


def run_benchmarks(
    # Parameters for scaling study
    # Each must be at <= n_components
    m_arr=[20, 24, 32, 64, 128, 256, 512, 1024, 2048],
    tol=0.1,
    k=0,
    # Parameters for detailed comparison
    detailed_comparison=True,
    comparison_params=None,
    random_state=2,
    run_gpu=False, # GPU control parameter
    print_samples_per_second = False
):
    """
    Comprehensive benchmarking function that can run both scaling studies and detailed comparisons.
    
    Args:
        m_arr: List of problem sizes (m) to test
        n_samples: Number of samples to generate
        tol: Tolerance for OMP algorithms
        k: Adjustment factor for n_nonzero_coefs
        detailed_comparison: Whether to run detailed comparison with error analysis
        comparison_params: Dict with specific parameters for detailed comparison
                          (overrides derived parameters if provided)
        random_state: Random seed for reproducibility
        run_gpu: Whether to run GPU algorithms (naive_gpu and v0_gpu)
    """
    
    execution_times = {
        "sklearn": [],
        "naive_cpu": [],
        "v0_cpu": []
    }
    
    # Add GPU entries only if GPU is enabled
    if run_gpu:
        execution_times["naive_gpu"] = []
        execution_times["v0_gpu"] = []
    
    # Set default comparison parameters if not provided
    if comparison_params is None:
        comparison_params = {
            'n_components': 100,
            'n_features': 100,
            'n_nonzero_coefs': 17,
            'n_samples': 50
        }

    n_components = comparison_params['n_components']
    n_features = comparison_params['n_features']
    n_nonzero_coefs = comparison_params['n_nonzero_coefs']
    current_n_samples = comparison_params['n_samples']    

    print("Settings used for the test:\n")
    # print(f"Number of Components: {n_components}")
    # print(f"Number of Features: {n_features}")
    print(f"Number of Nonzero Coefficients: {n_nonzero_coefs}")
    print(f"GPU algorithms enabled: {run_gpu}")
    print(f"Number of Samples: {current_n_samples}")
    
    for i, m in enumerate(m_arr):
        n_components = m
        n_features = 8*m
        print(f"\n{'='*50}")
        print(f"Testing problem size n_components = {n_components} and n_features = {n_features} ({i+1}/{len(m_arr)})")
        print(f"{'='*50}")

        # Generate data
        y, D, X = make_sparse_coded_signal(
            n_samples=current_n_samples,
            n_components=n_components,
            n_features=n_features,
            n_nonzero_coefs=n_nonzero_coefs,
            random_state=random_state
        )

        y = y.T
        # if detailed_comparison:
        #     # Add noise for detailed comparison
        #     y = y + np.random.randn(*y.shape) * 0.01

        omp_args = dict(
            tol=tol, 
            n_nonzero_coefs=n_nonzero_coefs-k, 
            precompute=False, 
            fit_intercept=True, 
            # normalize=True
        )

        # Run benchmarks and collect results
        results = {}
        
        # Sklearn
        omp = OrthogonalMatchingPursuit(**omp_args)
        with elapsed_timer() as elapsed:
            omp.fit(D.T, y)
        sklearn_time = elapsed()
        execution_times["sklearn"].append(sklearn_time)
        results['sklearn'] = omp.coef_
        print(f'Sklearn OMP runtime: {sklearn_time:.4f}')
        if print_samples_per_second:
            print(f'Sklearn OMP Samples per second: {current_n_samples / sklearn_time:.4f}\n')

        # Naive CPU
        with elapsed_timer() as elapsed:
            naive_cpu_result = run_omp(
                torch.as_tensor(D, device='cpu', dtype=torch.float), 
                torch.as_tensor(y, device='cpu', dtype=torch.float), 
                n_nonzero_coefs-k,
                tol=tol,
                # normalize=True, # Removed in 1.2
                fit_intercept=True,
                alg="naive"
            )
        naive_cpu_time = elapsed()
        execution_times["naive_cpu"].append(naive_cpu_time)
        results['naive_cpu'] = naive_cpu_result.numpy()
        print(f'Naive CPU runtime: {naive_cpu_time:.4f}')
        if print_samples_per_second:
            print(f'Naive CPU Samples per second: {current_n_samples / naive_cpu_time:.4f}\n')

        # V0 CPU
        with elapsed_timer() as elapsed:
            v0_cpu_result = run_omp(
                torch.as_tensor(D, device='cpu', dtype=torch.float), 
                torch.as_tensor(y, device='cpu', dtype=torch.float), 
                n_nonzero_coefs-k,
                tol=tol,
                # normalize=True,
                fit_intercept=True,
                alg="v0"
            )
        v0_cpu_time = elapsed()
        execution_times["v0_cpu"].append(v0_cpu_time)
        results['v0_cpu'] = v0_cpu_result.numpy()
        print(f'V0 CPU runtime: {v0_cpu_time:.4f}')
        if print_samples_per_second:
            print(f'V0 CPU Samples per second: {current_n_samples / v0_cpu_time:.4f}')

        # GPU algorithms (conditional)
        if run_gpu:
            # Naive GPU
            print('Running Naive GPU OMP...')
            with elapsed_timer() as elapsed:
                naive_gpu_result = gpu_transfer_and_alg(D, y, "naive", n_nonzero_coefs)
            naive_gpu_time = elapsed()
            execution_times["naive_gpu"].append(naive_gpu_time)
            print(f'Naive GPU runtime: {naive_gpu_time:.4f}')
            if print_samples_per_second:
                print(f'Samples per second: {current_n_samples / naive_gpu_time:.4f}')

            # V0 GPU
            print('Running V0 GPU OMP...')
            with elapsed_timer() as elapsed:
                v0_gpu_result = gpu_transfer_and_alg(D, y, "v0", n_nonzero_coefs)
            v0_gpu_time = elapsed()
            execution_times["v0_gpu"].append(v0_gpu_time)
            print(f'V0 GPU runtime: {v0_gpu_time:.4f}')
            if print_samples_per_second:
                print(f'Samples per second: {current_n_samples / v0_gpu_time:.4f}')
        else:
            print('Skipping GPU algorithms (run_gpu=False)')

        # Detailed comparison and error analysis (only for first iteration if enabled)
        if detailed_comparison:
            print(f"\n{'='*30} RECONSTRUCTION ERROR {'='*30}")
            
            # Print sparsity patterns
            # print(f"V0 CPU nonzero pattern: {v0_cpu_result.numpy().nonzero()}")
            # print(f"V0 CPU result shape: {v0_cpu_result.shape}")
            
            # Reconstruction error analysis
            sklearn_reconstruction_error = (np.linalg.norm(y[..., None] - D @ results['sklearn'][..., None], ord=2, axis=-2).squeeze(-1) ** 2).max()
            v0_reconstruction_error = (np.linalg.norm(y[..., None] - D @ results['v0_cpu'][..., None], ord=2, axis=-2).squeeze(-1) ** 2).max()
            naive_reconstruction_error = (np.linalg.norm(y[..., None] - D @ results['naive_cpu'][..., None], ord=2, axis=-2).squeeze(-1) ** 2).max()
            
            print(f"Sklearn reconstruction error: {sklearn_reconstruction_error:.6e}")
            print(f"V0 CPU reconstruction error: {v0_reconstruction_error:.6e}")
            print(f"Naive CPU reconstruction error: {naive_reconstruction_error:.6e}")
            
            print(f"{'='*80}")

    return execution_times


if __name__ == "__main__":
    run_benchmarks()