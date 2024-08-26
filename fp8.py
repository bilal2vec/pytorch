import time
import numpy as np

import torch

bmnks = [(2, 8192, 6144, 4096), (2, 8192, 4096, 4096), (2, 8192, 14336, 4096), (2, 8192, 4096, 14336)]

problems = {True: {}, False: {}}
reference = {True: {}, False: {}}

for _ in range(20):
    for (b, m, n, k) in bmnks:
        x = torch.ones((m, k), dtype=torch.float8_e4m3fn, device='cuda')
        w = torch.ones((n, k), dtype=torch.float8_e4m3fn, device='cuda')
        scale = torch.tensor([1.0], dtype=torch.float32, device='cuda')

        for tunable_op_enabled in [False, True]:
            torch.cuda.tunable.enable(tunable_op_enabled)           

            for _ in range(5):
                z = torch._scaled_mm(x, w.T, scale_a=scale, scale_b=scale)

            torch.cuda.synchronize()

            times = []
            for _ in range(10):
                start = time.time()
                z = torch._scaled_mm(x, w.T, scale_a=scale, scale_b=scale)
                torch.cuda.synchronize()
                times.append(time.time() - start)

            problem = f"{m}x{n}x{k}"
            s = sum(times) / len(times)
            stddev = np.std(np.array(times) / 1e-6)
            us = s / 1e-6
            problems[tunable_op_enabled][problem] = us
            reference[tunable_op_enabled][problem] = z

            #print(z)
            if tunable_op_enabled:
                assert(torch.allclose(reference[False][problem].to(torch.float32), reference[True][problem].to(torch.float32)))
                speedup = problems[False][problem] / problems[True][problem]

            power = torch.cuda.power_draw() / 1000.0
            tflops = 2 * m * n * k / s / 1e12

            print(f'{problem}: \t {us:.2f} (+-{stddev:.2f})  tflops: {tflops:.2f} power: {power:.2f} speedup: {speedup if tunable_op_enabled else 0}')
