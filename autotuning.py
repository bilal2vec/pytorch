# From https://gist.github.com/IvanYashchuk/46535407409b637b42fc3042f691976f
import logging
import itertools
#logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

import os
os.environ["CUBLASLT_WORKSPACE_SIZE"] = "32" # 32 MB workspace for cuBLASLt

import torch

from torch.utils.benchmark import Timer, Compare

linear_input_shapes = [((2, 8192, 4096), (4096, 6144)), ((2, 8192, 4096), (4096, 4096)), ((2, 8192, 4096), (4096, 14336)), ((2, 8192, 14336), (14336, 4096))]
import nvmath

n = 8
preferences = nvmath.linalg.advanced.MatmulPlanPreferences(limit=8)

results = []
speedups = []

def get_timings(x_shape, w_shape, quiet=False):
    x = torch.randn(x_shape, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(w_shape, device="cuda", dtype=torch.bfloat16)
    w_transposed = w.mT

    # (2, 8192, 6144)
    out = torch.randn((*x.shape[:-1], w.shape[-1]), device="cuda", dtype=torch.bfloat16)

    label = f"({x_shape}, {w_shape})"

    fw_pytorch = Timer(
        stmt='torch.matmul(x, w)',
        globals={'x': x, "w": w},
        label = label,
        sub_label="FW",
        description="PyTorch")

    bw_dx_batched_pytorch = Timer(
        stmt='torch.matmul(out, w.mT)',
        globals={'out': out, "w": w},
        label = label,
        sub_label="BW dx (batched)",
        description="PyTorch")

    bw_dx_folded_pytorch = Timer(
        stmt='torch.matmul(out.reshape(-1, out.shape[-1]), w.mT)',
        globals={'out': out, "w": w},
        label = label,
        sub_label="BW dx (folded)",
        description="PyTorch")

    bw_dw_pytorch = Timer(
        stmt='torch.matmul(out.reshape(-1, out.shape[-1]).mT, x.reshape(-1, x.shape[-1]))',
        globals={'out': out, "x": x},
        label = label,
        sub_label="BW dw",
        description="PyTorch")

    mm_forward = nvmath.linalg.advanced.Matmul(x, w)
    mm_forward.plan(preferences=preferences)
    mm_forward.autotune(iterations=n)
    
    fw_cublaslt = Timer(
        stmt='mm_forward.execute()',
        globals={"mm_forward": mm_forward},
        label = label,
        sub_label="FW",
        description="nvmath (cuBLASTLt)")

    mm_backward_dx_batched = nvmath.linalg.advanced.Matmul(out, w_transposed)
    mm_backward_dx_batched.plan(preferences=preferences)
    mm_backward_dx_batched.autotune(iterations=n)

    bw_dx_batched_cublaslt = Timer(
        stmt='mm_backward_dx_batched.execute()',
        globals={"mm_backward_dx_batched": mm_backward_dx_batched},
        label = label,
        sub_label="BW dx (batched)",
        description="nvmath (cuBLASTLt)")

    mm_backward_dx_folded = nvmath.linalg.advanced.Matmul(out.reshape(-1, out.shape[-1]), w_transposed)
    mm_backward_dx_folded.plan(preferences=preferences)
    mm_backward_dx_folded.autotune(iterations=n)

    bw_dx_folded_cublaslt = Timer(
        stmt='mm_backward_dx_folded.execute()',
        globals={"mm_backward_dx_folded": mm_backward_dx_folded},
        label = label,
        sub_label="BW dx (folded)",
        description="nvmath (cuBLASTLt)")

    mm_backward_dw = nvmath.linalg.advanced.Matmul(out.reshape(-1, out.shape[-1]).mT, x.reshape(-1, x.shape[-1]))
    mm_backward_dw.plan(preferences=preferences)
    mm_backward_dw.autotune(iterations=n)

    bw_dw_cublaslt = Timer(
        stmt='mm_backward_dw.execute()',
        globals={"mm_backward_dw": mm_backward_dw},
        label = label,
        sub_label="BW dw",
        description="nvmath (cuBLASTLt)")

    mm_forward.logger.setLevel(logging.WARNING)
    mm_backward_dx_batched.logger.setLevel(logging.WARNING)
    mm_backward_dx_folded.logger.setLevel(logging.WARNING)
    mm_backward_dw.logger.setLevel(logging.WARNING)

    local_results = []
    pytorch_timers = [fw_pytorch, bw_dx_batched_pytorch, bw_dx_folded_pytorch, bw_dw_pytorch]
    cublaslt_timers = [fw_cublaslt, bw_dx_batched_cublaslt, bw_dx_folded_cublaslt, bw_dw_cublaslt]
    timers = list(itertools.chain(*zip(pytorch_timers, cublaslt_timers)))
    if not quiet:
        for timer in timers: 
            out = timer.blocked_autorange(min_run_time=1)
            print(f'{torch.cuda.power_draw() / 1000.0:.2f}')
            results.append(out)
            local_results.append(results[-1])

        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        for baseline, candidate in batch(local_results, 2):
            speedup = baseline.median / candidate.median
            speedups.append({f"{label} {baseline.sub_label}": f"{speedup:.2f}x"})

        for mm in (mm_forward, mm_backward_dx_batched, mm_backward_dx_folded, mm_backward_dw):
            mm.free()

    pass

for i in range(2):
    for x_shape, w_shape in linear_input_shapes:
        get_timings(x_shape, w_shape, quiet=i==0)

compare = Compare(results)
compare.colorize(rowwise=True)
compare.print()

from pprint import pprint
pprint(speedups)
