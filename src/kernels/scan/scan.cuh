#pragma once

#include "core/common.h"

#define SECTION_SIZE 1024

#define CFACTOR 4
namespace cuda_kernels::scan
{
// constexpr unsigned int CFACTOR = 4;

__global__ void kogge_stone_scan(float *X, float *Y, unsigned int N);

__global__ void brent_kung_scan(float *X, float *Y, unsigned int N);


__device__ void brent_kung_scan_block(float *data, unsigned int tx, unsigned raw_stride);
__device__ void scan_block(float *data, unsigned int c_factor);

__global__ void coarsened_scan(float *X, float *Y, unsigned int N);
__global__ void all_scan(float *X, float *Y, unsigned int N);

__global__ void scan_phase1(float *X, float *Y, unsigned int N, float *block_sums);
__global__ void scan_phase2(unsigned int N, float *block_sums);
__global__ void scan_phase3(float *Y, unsigned int N, float *block_sums);
}