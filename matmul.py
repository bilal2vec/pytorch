import time
import numpy as np
from itertools import product

import torch


sizes = [4096, 6144, 8192, 14336]
batch_sizes = [2]

problems = {}
percent_speedups = []

for i in range(2):
    for backend in ["cublas", "cublaslt"]:
        problems[backend] = {}
        torch.backends.cuda.preferred_blas_library(backend)

        for (b, m, n, k) in product(batch_sizes, sizes, sizes, sizes):
            x = torch.ones((b, m, k), dtype=torch.bfloat16, device='cuda')
            y = torch.ones((b, k, n), dtype=torch.bfloat16, device='cuda')

            for _ in range(5):
                z = torch.matmul(x, y)

            torch.cuda.synchronize()

            times = []

            for _ in range(10):
                start = time.time()
                z = torch.matmul(x, y)
                torch.cuda.synchronize()
                times.append(time.time() - start)

            problem = f"{b}x{m}x{n}x{k}"
            s = sum(times) / len(times)
            stddev = np.std(np.array(times) / 1e-6)
            us = s / 1e-6
            problems[backend][problem] = us

            power = torch.cuda.power_draw() / 1000.0
            tflops = 2 * b * m * n * k / s / 1e12

            if backend == 'cublaslt':
                cublas_time = problems['cublas'][problem]
                percent_speedup = (cublas_time - us) / cublas_time * 100
                percent_speedups.append(percent_speedup)
            else:
                cublas_time = us
                percent_speedup = 0
    
            print(f'{problem}: \t {us:.2f} (+-{stddev:.2f}) % speedup: {percent_speedup:.2f} ({cublas_time - us:.2f} us) tflops: {tflops:.2f} power: {power:.2f}')

print(f'avg percent speedup: {sum(percent_speedups) / len(percent_speedups):.2f}')
