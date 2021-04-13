import cython
# import numpy as np
cimport numpy as np

from cpython cimport PyCapsule_GetPointer
from scipy.linalg.cython_blas cimport idamax, isamax, daxpy
cimport scipy.linalg.cython_lapack as lapack
ctypedef np.float64_t REAL_t
ctypedef np.int64_t  INT_t

# ctypedef void (*idamax_ptr) (const int *n, const double *dx, const int *incx) nogil
# cdef idamax_ptr idamax=<idamax_ptr>PyCapsule_GetPointer(LA.blas.idamax._cpointer, NULL)  # A := alpha*x*y.T + A

# TODO: cdef/cpdef, fused functions/types, specialization: https://cython.readthedocs.io/en/latest/src/userguide/fusedtypes.html#type-checking-specializations

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void update_projections_blast(double[:, :] projections,
                             double[:, :] D_mybest, double[:] coefs) nogil:
    cdef Py_ssize_t B = projections.shape[0]
    cdef int N = projections.shape[1]
    cdef int incy = projections.strides[1] // sizeof(double)     # Stride between elements.
    cdef int incx = D_mybest.strides[1] // sizeof(double)  # Stride between elements.
    cdef Py_ssize_t i
    # TODO: Loop unrolling?
    for i from 0 <= i < B:
        daxpy(&N, &coefs[i], &D_mybest[i, 0], &incx, &projections[i, 0], &incy)  # np.argmax(np.abs(projections[i]))

ctypedef fused proj_t:
    double
    float





@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void argmax_blast(proj_t[:, :] projections,
                 long long[:] output) nogil:
    # http://conference.scipy.org/static/wiki/seljebotn_cython.pdf
    # https://apprize.best/python/cython/3.html
    cdef Py_ssize_t B = projections.shape[0]
    cdef int N = projections.shape[1]
    cdef int incx = projections.strides[1] // sizeof(proj_t)  # Stride between elements.
    cdef Py_ssize_t i
    # TODO: Just create a preprocessor directive to generate all the specializations.
    for i from 0 <= i < B:
        if proj_t is double:  # One C-function is created for each of these specializations :) (see argmax_blast.__signatures__)
            output[i] = idamax(&N, &projections[i, 0], &incx) - 1
        elif proj_t is float:
            output[i] = isamax(&N, &projections[i, 0], &incx) - 1


@cython.boundscheck(False)
@cython.wraparound(False)
def argmax_blas(np.ndarray[np.float64_t, ndim=2] projections,
                np.ndarray[np.int64_t, ndim=1] output):
    # http://conference.scipy.org/static/wiki/seljebotn_cython.pdf
    cdef Py_ssize_t B = projections.shape[0]
    cdef int N = projections.shape[1]
    cdef int incx = projections.strides[1] // sizeof(np.float64_t)  # Stride between elements.
    cdef Py_ssize_t skip = projections.strides[0] // sizeof(np.float64_t)  # Second stride.
    cdef Py_ssize_t i
    cdef REAL_t *_projections = <REAL_t *>(np.PyArray_DATA(projections))

    with nogil:
        for i from 0 <= i < B:
            output[i] = <np.int64_t> ( idamax(&N, _projections + i*skip, &incx) - 1 )

def get_max_projections_blas(_projections, _output):
    #  BLAS is even faster! :O
    # Maybe remove overhead with Cython? https://stackoverflow.com/questions/44710838/calling-blas-lapack-directly-using-the-scipy-interface-and-cython, https://yiyibooks.cn/sorakunnn/scipy-1.0.0/scipy-1.0.0/linalg.cython_blas.html
    # func = 'i' + scipy.linalg.blas.find_best_blas_type(dtype=projections.dtype)[0] + 'amax'
    # func = getattr(scipy.linalg.blas, func)
    # B = projections.shape[0]
    cdef int B = _projections.shape[0]
    cdef int N = _projections.shape[1]
    cdef int incx = 1
    cdef REAL_t *projections = <REAL_t *>(np.PyArray_DATA(_projections))
    cdef INT_t *output = <INT_t *>(np.PyArray_DATA(_output))

    with nogil:
        for i in range(B):
            output[i] = idamax(&N, &(projections[i]), &incx)