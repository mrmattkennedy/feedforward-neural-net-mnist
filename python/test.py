import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
import skcuda.misc as misc
import time
import pdb


linalg.init()

start = time.time()
for _ in range(5):
    a = np.asarray(np.random.rand(60000, 784), np.float32)
    b = np.asarray(np.random.rand(784, 600), np.float32)
    d = np.asarray(np.random.rand(600, 500), np.float32)
    np.dot(np.dot(a, b), d)
print(time.time() - start)

start = time.time()
a = np.asarray(np.random.rand(60000, 784), np.float32)
b = np.asarray(np.random.rand(784, 600), np.float32)
d = np.asarray(np.random.rand(600, 500), np.float32)
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
d_gpu = gpuarray.to_gpu(d)
for _ in range(20):
    c = linalg.dot(a_gpu, b_gpu)
    e = linalg.dot(c, d_gpu)
print(time.time() - start)

