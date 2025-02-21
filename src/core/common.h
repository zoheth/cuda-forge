#pragma once

#include <cuda/std/span>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/memory.h>

#include <stdexcept>

#define CUDA_CHECK(err)                                                  \
	do                                                                   \
	{                                                                    \
		cudaError_t err_ = (err);                                        \
		if (err_ != cudaSuccess)                                         \
		{                                                                \
			throw std::runtime_error(                                    \
			    std::string("CUDA error: ") + cudaGetErrorString(err_) + \
			    " at " + __FILE__ + ":" + std::to_string(__LINE__));     \
		}                                                                \
	} while (0)

#define CUDA_CHECK_NOTHROW(err)                                    \
	do                                                             \
	{                                                              \
		cudaError_t err_ = (err);                                  \
		if (err_ != cudaSuccess)                                   \
		{                                                          \
			fprintf(stderr, "CUDA error: %s at %s:%d\n",           \
			        cudaGetErrorString(err_), __FILE__, __LINE__); \
		}                                                          \
	} while (0)

#define MAKE_CONST_SPAN(vec)                                                            \
	cuda::std::span<typename std::remove_reference_t<decltype(vec)>::value_type const>( \
	    thrust::raw_pointer_cast((vec).data()),                                         \
	    (vec).size())

#define MAKE_SPAN(vec)                                                            \
	cuda::std::span<typename std::remove_reference_t<decltype(vec)>::value_type>( \
	    thrust::raw_pointer_cast((vec).data()),                                   \
	    (vec).size())


#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
