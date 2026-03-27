__version__ = "0.1.0"

from .omp import run_omp, omp_v0, omp_v0_blas

__all__ = ["run_omp", "omp_v0", "omp_v0_blas"]

try:
    from .sklearn_compat import BatchedOrthogonalMatchingPursuit
    __all__.append("BatchedOrthogonalMatchingPursuit")
except ImportError:
    pass
