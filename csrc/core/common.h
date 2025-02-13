#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

// 通用宏定义
#define CUDA_CHECK(call)                                         \
	{                                                            \
		cudaError_t err = call;                                  \
		if (err != cudaSuccess)                                  \
		{                                                        \
			printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
			       cudaGetErrorString(err));                     \
			exit(1);                                             \
		}                                                        \
	}

#define NVTX_RANGE(name)  \
	nvtxRangePushA(name); \
	auto start = std::chrono::high_resolution_clock::now()

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

namespace cuda_kernels
{
namespace utils
{

template <typename T>
void deviceMalloc(T **ptr, size_t size)
{
	CUDA_CHECK(cudaMalloc(ptr, size * sizeof(T)));
}

}        // namespace utils
}        // namespace cuda_kernels