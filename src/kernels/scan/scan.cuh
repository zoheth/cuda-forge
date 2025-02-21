#pragma once

#include "core/common.h"

#define SECTION_SIZE 1024

namespace cuda_kernels::scan
{
__global__ void kogge_stone_scan(float *X, float *Y, unsigned int N);

__global__ void brent_kung_scan(float *X, float *Y, unsigned int N);

__device__ void scan_block(float *data, unsigned int size, unsigned int c_factor);

__global__ void scan_phase1(float *X, float *Y, unsigned int N, unsigned int *block_sums);
__global__ void scan_phase2(unsigned int N, unsigned int *block_sums);
__global__ void scan_phase3(float *Y, unsigned int N, unsigned int *block_sums);
}