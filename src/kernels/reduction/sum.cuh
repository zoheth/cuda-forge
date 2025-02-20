#pragma once

#include "core/common.h"

#define BLOCK_DIM 1024

namespace cuda_kernels::reduction
{
__global__ void simple_sum_kernel(float *input, float *output);

__global__ void simple_sum_kernel_1(float *input, float *output);

__global__ void shared_sum_kernel(float *input, float *output);

__global__ void segmented_sum_kernel(float *input, float *output);
}