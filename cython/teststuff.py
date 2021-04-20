import numpy as np
from test_stuff import argmax_blast

output = np.zeros((9000000,), dtype=np.int64)
output2 = np.zeros((9000000,), dtype=np.int64)
proj = np.random.randn(9000000, 20).astype(np.float32)

# argmax_blas(np.ascontiguousarray(proj.T).T, output)
argmax_blast(proj, output)
argmax_blast(np.ascontiguousarray(proj.T).T, output2)
print(output, output2)
print(np.argmax(np.abs(proj), 1))
print(proj.T.strides)
print(np.ascontiguousarray(proj.T).T.strides)

print(np.max(np.abs(output - np.argmax(np.abs(proj), 1))))