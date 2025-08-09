import numpy as np
from test import argmax_blast, update_D_mybest_blast, posv
import pickle

# python setup.py build_ext --inplace ; cp test.cp37-win_amd64.pyd .. ; python ../test_omp.py
# kernprof -l -v ../test_omp.py


# Test update_D_mybest
def update_D_mybest_fast(temp_F_k_k, XTX, maxindices, A, x, D_mybest):
    # D_mybest[...] = -temp_F_k_k * (A @ x[:, :, None]).squeeze(-1)
    # ^ This is a parallelized (faster) version of the first line in the loop below.
    for i in range(temp_F_k_k.shape[0]):
        D_mybest[i] = -temp_F_k_k[i] * (A[i] @ x[i, :, None]).squeeze(-1)  # dgemv
        D_mybest[i] = temp_F_k_k[i] * XTX[maxindices[i]] + D_mybest[i]  # daxpy
    return D_mybest

with open('update_D_mybest_data.pkl', 'rb') as f:
    temp_F_k_k, XTX, maxindices, A, x, D_mybest = pickle.load(f)
    # print('true strides?:', A.transpose([1, 2, 0])[0].strides)
    D_mybest_numpy = update_D_mybest_fast(temp_F_k_k, XTX, maxindices, A.transpose([1, 2, 0]), x, D_mybest.copy())
    D_mybest_blas = D_mybest.copy()
    A = A.transpose([1, 0, 2]).copy().transpose([0, 2, 1])
    print('strides:', A[0].strides, A.shape, D_mybest_blas.shape, D_mybest_blas.strides)
    print(A.shape, D_mybest_blas.shape)
    update_D_mybest_blast(temp_F_k_k, XTX, maxindices, A, x, D_mybest_blas)

print('Change in D_mybest:', np.max(np.abs(D_mybest_numpy-D_mybest)))
print('ERROR IN update_D_mybest:', np.max(np.abs(D_mybest_numpy-D_mybest_blas)))

exit(0)


# Test posv
with open('sposv.pkl', 'rb') as f:
    As, ys = pickle.load(f)
    ys = ys[:, :As.shape[1]]  # Accidentally saved one too many.

ys = np.ascontiguousarray(ys[:, :2]) # -1:0:-2
As = As[:, :2, :2]
print(ys.shape, ys.strides, ys[0].strides)
xests = np.linalg.solve(As, ys)
print(xests.shape)
print(As.shape)

posv(As, ys)
xests_posv = ys

print(np.max(np.abs(xests-xests_posv)))


# Test argmax
output = np.zeros((200,), dtype=np.int64)
output2 = np.zeros((200,), dtype=np.int64)
proj = np.random.randn(200, 5).astype(np.float32)

# argmax_blas(np.ascontiguousarray(proj.T).T, output)
argmax_blast(proj, output)
argmax_blast(np.ascontiguousarray(proj.T).T, output2)
print(output, output2)
print(np.argmax(np.abs(proj), 1))
print(proj.T.strides)
print(np.ascontiguousarray(proj.T).T.strides)

print('ERROR IN ARGMAX: ', np.max(np.abs(output - np.argmax(np.abs(proj), 1))))
