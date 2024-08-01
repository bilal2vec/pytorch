import time
import numpy as np
from itertools import product

import torch


sizes = [2**(4*i) for i in range(4)] + [8192]

problems = {}
percent_speedups = []

for backend in ["cublas", "cublaslt"]:
    problems[backend] = {}
    torch.backends.cuda.preferred_blas_library(backend)

    for (m, n, k) in product(sizes, sizes, sizes):
        x = torch.ones((m, k), dtype=torch.float32, device='cuda')
        y = torch.ones((k, n), dtype=torch.float32, device='cuda')

        for _ in range(5):
            z = torch.matmul(x, y)

        torch.cuda.synchronize()

        times = []

        for _ in range(100):
            start = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            times.append(time.time() - start)

        problem = f"{m}x{n}x{k}"
        s = sum(times) / len(times)
        stddev = np.std(np.array(times) / 1e-6)
        us = s / 1e-6
        problems[backend][problem] = us

        if backend == 'cublaslt':
            tflops = 2 * m * n * k / s / 1e12

            cublas_time = problems['cublas'][problem]
            percent_speedup = (cublas_time - us) / cublas_time * 100
            percent_speedups.append(percent_speedup)

            print(f'{problem}: \t {us:.2f} (+-{stddev:.2f}) % speedup: {percent_speedup:.2f} tflops: {tflops:.2f}')

print(f'avg percent speedup: {sum(percent_speedups) / len(percent_speedups):.2f}')
