#include "adds.cuh"

#include "core/common.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace cuda_kernels::elementwise
{
__global__ void add_f32_kernel(float *a, float *b, float *c, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
		c[idx] = a[idx] + b[idx];
}
__global__ void add_f32x4_kernel(float *a, float *b, float *c, int N)
{
	int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
	if (idx < N)
	{
		float4 reg_a = FLOAT4(a[idx]);
		float4 reg_b = FLOAT4(b[idx]);
		float4 reg_c;
		reg_c.x = reg_a.x + reg_b.x;
		reg_c.y = reg_a.y + reg_b.y;
		reg_c.z = reg_a.z + reg_b.z;
		reg_c.w = reg_a.w + reg_b.w;
		FLOAT4(c[idx]) = reg_c;
	}
}

__global__ void add_f16_kernel(half *a, half *b, half *c, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		c[idx] = __hadd(a[idx], b[idx]);
	}
}

__global__ void add_f16x2_kernel(half *a, half *b, half *c, int N)
{
	int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	if (idx < N)
	{
		half2 reg_a = HALF2(a[idx]);
		half2 reg_b = HALF2(b[idx]);
		half2 reg_c;
		reg_c.x = __hadd(reg_a.x, reg_b.x);
		reg_c.y = __hadd(reg_a.y, reg_b.y);
		HALF2(c[idx]) = reg_c;
	}
}

__global__ void add_f16x8_pack_kernel(half *a, half *b, half *c, int N)
{
	int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);

	half pack_a[8], pack_b[8], pack_c[8];

	LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
	LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

	#pragma unroll
	for (int i=0; i<8; i+=2)
	{
		HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
	}

	if (idx + 7 < N)
	{
		LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
	}
}

}        // namespace cuda_kernels::elementwise