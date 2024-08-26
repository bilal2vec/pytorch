import time
import numpy as np

import torch

bmnks = [(2, 8192, 6144, 4096), (2, 8192, 4096, 4096), (2, 8192, 14336, 4096), (2, 8192, 4096, 14336)]

problems = {True: {}, False: {}}
reference = {True: {}, False: {}}

for _ in range(3):
    for (b, m, n, k) in bmnks:
        x = torch.ones((b, m, k), dtype=torch.bfloat16, device='cuda')
        w = torch.ones((k, n), dtype=torch.bfloat16, device='cuda')
        out = torch.ones((b, m, n), dtype=torch.bfloat16, device='cuda')

        for tunable_op_enabled in [False, True]:
            torch.cuda.tunable.enable(tunable_op_enabled)           

            permutations = [("fw_x_b", x, w), ("bw_dx_b", out, w.mT), ("bw_dx_f", out.reshape(-1, out.shape[-1]), w.mT), ("bw_dw_f", out.reshape(-1, out.shape[-1]).mT, x.reshape(-1, x.shape[-1]))]
            for (tag, x1, w1) in permutations:
                for _ in range(5):
                    z = torch.matmul(x1, w1)

                torch.cuda.synchronize()

                times = []
                for _ in range(10):
                    start = time.time()
                    z = torch.matmul(x1, w1)
                    torch.cuda.synchronize()
                    times.append(time.time() - start)

                problem = f"{b}x{m}x{n}x{k}_" + tag
                s = sum(times) / len(times)
                stddev = np.std(np.array(times) / 1e-6)
                us = s / 1e-6
                problems[tunable_op_enabled][problem] = us
                reference[tunable_op_enabled][problem] = z

                if tunable_op_enabled:
                    assert(torch.allclose(reference[False][problem], reference[True][problem]))
                    speedup = problems[False][problem] / problems[True][problem]

                power = torch.cuda.power_draw() / 1000.0
                tflops = 2 * b * m * n * k / s / 1e12

                print(f'{problem}: \t {us:.2f} (+-{stddev:.2f})  tflops: {tflops:.2f} power: {power:.2f} speedup: {speedup if tunable_op_enabled else 0}')
