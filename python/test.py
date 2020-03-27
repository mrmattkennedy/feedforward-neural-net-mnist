import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
import skcuda.misc as misc
import time
import pdb

"""
linalg.init()
a = np.asarray(np.random.rand(1000, 10000), np.float32)
b = np.asarray(np.random.rand(10000, 2000), np.float32)
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)

start = time.time()
for _ in range(1):
    np.dot(a, b)
print(time.time() - start)

start = time.time()
for _ in range(50):
    pdb.set_trace()
    linalg.dot(a_gpu, b_gpu)
print(time.time() - start)
"""
temp = np.array([1, 2, 3])
print(np.tile(temp, 2).reshape((-1,temp.shape[0])))
