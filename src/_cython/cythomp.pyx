import cython
from cython cimport view
# import numpy as np
cimport numpy as np

from cpython cimport PyCapsule_GetPointer
from scipy.linalg.cython_blas cimport idamax, isamax, daxpy, dgemv, dtrmv, dcopy
from scipy.linalg.cython_lapack cimport dposv, dppsv, sppsv
from libc.string cimport memcpy
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
    cdef int incy = projections[0].strides[1] // sizeof(double)     # Stride between elements.
    cdef int incx = D_mybest[0].strides[1] // sizeof(double)  # Stride between elements.
    cdef Py_ssize_t i
    # TODO: Loop unrolling?
    for i from 0 <= i < B:
        daxpy(&N, &coefs[i], &D_mybest[i, 0], &incx, &projections[i, 0], &incy)

ctypedef fused proj_t:
    double
    float


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void update_D_mybest_blast(double[:] temp_F_k_k, double[:, :] XTX,
                          long long[:] maxindices, double[:, :, :] A,
                          double[:, :] x, double[:, :] D_mybest) nogil:
    cdef Py_ssize_t B = A.shape[0]  # Batch size
    cdef int N = A.shape[1]  # m A.shape[2]
    cdef int k = A.shape[2]  # n
    # cdef int N = A.shape[2]  # n
    # cdef int k = A.shape[1]  # m A.shape[2]
    # cdef int ldaA = (A[0].strides[0]) // sizeof(double)  # Stride in A.
    cdef int ldaA = 2048 # Stride in A.
    # cdef int ldaA = A[0].strides[0] // sizeof(double)  # Stride in A.
    cdef int incx = x[0].strides[1] // sizeof(double)  # Stride between elements.
    cdef int incy = D_mybest[0].strides[0] // sizeof(double)  # Stride between elements.
    cdef int incXTX = XTX.strides[1] // sizeof(double)  # Stride between elements.
    cdef char trans = 'T'
    cdef double minus_temp_F_k_k
    # Can we use omp parallel for here?
    for i from 0 <= i < B:
        # dcopy(&N, &XTX[maxindices[i], 0], &incXTX, &D_mybest[i, 0], &incy)
        # ^ D_mybest[i] = XTX[maxindices[i]]

        minus_temp_F_k_k = -temp_F_k_k[i]  # Great
        dgemv(alpha=&minus_temp_F_k_k, beta=&temp_F_k_k[i],
              a=&A[i, 0, 0], n=&k, m=&N, lda=&ldaA,
              x=&x[i, 0], incx=&incx,
              y=&D_mybest[i, 0], incy=&incy,
              trans=&trans)
        # ^ D_mybest[i] = temp_F_k_k[i] * D_mybest[i] - temp_F_k_k[i] * (A[i] @ x[i, :, None]).squeeze(-1)


#@cython.boundscheck(False)
#@cython.wraparound(False)
#cpdef void trmv(double[:] minus_temp_F_k_k, double[:, :, :] F, double[:, :] D_mybest_maxindices) nogil:
#    cdef Py_ssize_t B = F.shape[0]  # Batch size
#    cdef char uplo = 'U'
#    cdef char trans = 'N'

#    cdef Py_ssize_t B = F.shape[0]  # Batch size
#    cdef int k = F.shape[2]  # n
#    cdef int ldaA = (F.strides[0] * B) // sizeof(double)  # Stride in A.
#    cdef int incx = x.strides[1] // sizeof(double)  # Stride between elements.
#    cdef int incy = D_mybest.strides[1] // sizeof(double)  # Stride between elements.
#    cdef int incX = XTX.strides[1] // sizeof(double)  # Stride between elements.

# Xt, r, AT[:, k, :], sets[k, :]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void ppsv(proj_t[:, :] As,
           proj_t[:, :, :] ys) nogil:
    # Works not for strided array I think. And please do not give a negative-stride
    cdef Py_ssize_t B = ys.shape[0]  # Batch size
    cdef int N = ys.shape[1]
    cdef int nrhs = ys.shape[2]
    cdef int info = 0  # Just discard any error signals ;)
    cdef char uplo = 85 # 'U'
    # cdef int ldb = ys[0].strides[0] // sizeof(double)

    for i from 0 <= i < B:
        if proj_t is double:  # One C-function is created for each of these specializations :) (see argmax_blast.__signatures__)
            dppsv(&uplo, &N, &nrhs, &As[i, 0], &ys[i, 0, 0], &N, &info)
        elif proj_t is float:
            sppsv(&uplo, &N, &nrhs, &As[i, 0], &ys[i, 0, 0], &N, &info)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void argmax_blast(proj_t[:, :] projections,
                 long long[:] output) nogil:
    # TODO: Numpy has its own indexing data-type - this would be the appropriate output, and may even be faster.
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
cpdef void project_argmax(proj_t[:, :] XT, proj_t[:, :] r,
                        proj_t[:, :] projections,
                        long long[:] output) nogil:
    # TODO: Numpy has its own indexing data-type - this would be the appropriate output, and may even be faster.
    # http://conference.scipy.org/static/wiki/seljebotn_cython.pdf
    # https://apprize.best/python/cython/3.html
    cdef Py_ssize_t B = projections.shape[0]
    cdef int M = XT.shape[1]
    cdef int N = XT.shape[0]
    cdef int incx = projections.strides[1] // sizeof(proj_t)  # Stride between elements.
    cdef Py_ssize_t i
    cdef proj_t alpha = 1
    cdef proj_t beta = 0
    cdef int one = 1
    cdef char trans = 'N'
    # TODO: Just create a preprocessor directive to generate all the specializations.
    for i from 0 <= i < B:
        if proj_t is double:  # One C-function is created for each of these specializations :) (see argmax_blast.__signatures__)
            dgemv(alpha=&alpha, beta=&beta,
            a=&XT[0, 0], n=&M, m=&N, lda=&N,
            x=&r[i, 0], incx=&one,
            y=&projections[i, 0], incy=&one,
            trans=&trans)
            output[i] = idamax(&N, &projections[i, 0], &one) - 1
        elif proj_t is float:
            output[i] = isamax(&N, &projections[i, 0], &one) - 1


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


#def get_max_projections_blas(_projections, _output):
    #  BLAS is even faster! :O
    # Maybe remove overhead with Cython? https://stackoverflow.com/questions/44710838/calling-blas-lapack-directly-using-the-scipy-interface-and-cython, https://yiyibooks.cn/sorakunnn/scipy-1.0.0/scipy-1.0.0/linalg.cython_blas.html
    # func = 'i' + scipy.linalg.blas.find_best_blas_type(dtype=projections.dtype)[0] + 'amax'
    # func = getattr(scipy.linalg.blas, func)
    # B = projections.shape[0]
#    cdef int B = _projections.shape[0]
#    cdef int N = _projections.shape[1]
#    cdef int incx = 1
#    cdef REAL_t *projections = <REAL_t *>(np.PyArray_DATA(_projections))
#    cdef INT_t *output = <INT_t *>(np.PyArray_DATA(_output))

#    with nogil:
#        for i in range(B):
#            output[i] = idamax(&N, &(projections[i]), &incx)