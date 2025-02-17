#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace cuda_kernels::elementwise
{

// FP32 kernels
__global__ void add_f32_kernel(float *a, float *b, float *c, int N);
__global__ void add_f32x4_kernel(float *a, float *b, float *c, int N);

// FP16 kernels
__global__ void add_f16_kernel(half *a, half *b, half *c, int N);
__global__ void add_f16x2_kernel(half *a, half *b, half *c, int N);
__global__ void add_f16x8_kernel(half *a, half *b, half *c, int N);
__global__ void add_f16x8_pack_kernel(half *a, half *b, half *c, int N);

}        // namespace cuda_kernels::elementwise