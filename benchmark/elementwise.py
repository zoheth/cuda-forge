import torch
import time
import os
from torch.utils.cpp_extension import load
from typing import Optional
from functools import partial

torch.set_grad_enabled(False)

project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_root)

# Load the CUDA kernel as a python module
lib = load(name='elementwise_lib',
           sources=["csrc/kernels/elementwise/adds.cu",
                    "csrc/torch/binding.cu"],
           extra_include_paths=[
               os.path.join(parent_dir, "csrc"),  # 添加csrc目录到包含路径
               os.path.join(parent_dir, "csrc/kernels")
           ],
           extra_cuda_cflags=[
               "-O3",
               "-U__CUDA_NO_HALF_OPERATORS__",
               "-U__CUDA_NO_HALF_CONVERSIONS__",
               "-U__CUDA_NO_HALF2_OPERATORS__",
               "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
               "--expt-relaxed-constexpr",
               "--expt-extended-lambda",
               "--use_fast_math",
           ],
           extra_cflags=['-std=c++17'])

def run_benchmark(perf_func: callable, a: torch.Tensor, b: torch.Tensor, tag: str,
                  out: Optional[torch.Tensor] = None, warmup: int = 10,
                  iters: int = 1000, show_all: bool = False):
    # torch.dot vs custom dot_prod kernel
    if out is not None:
        out.fill_(0)
        # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all: print(out)
    return out, mean_time

Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

for (S, K) in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    a = torch.randn((S, K)).cuda().float().contiguous()
    b = torch.randn((S, K)).cuda().float().contiguous()
    c = torch.zeros_like(a).cuda().float().contiguous()
    run_benchmark(lib.elementwise_add_f32,   a, b, "f32",   c)
    run_benchmark(partial(torch.add, out=c), a, b, "f32_th")