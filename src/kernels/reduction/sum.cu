#include "sum.cuh"

namespace cuda_kernels::reduction
{

__global__ void simple_sum_kernel(float *input, float *output)
{
	unsigned int i = 2 * threadIdx.x;
	for (unsigned int stride = 1; stride <= blockDim.x; stride <<= 1)
	{
		if (threadIdx.x % stride == 0)
		{
			input[i] += input[i + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0)
	{
		*output = input[0];
	}
}

__global__ void simple_sum_kernel_1(float *input, float *output)
{
	unsigned int i = threadIdx.x;
	for (unsigned int stride = blockDim.x; stride > 0; stride >>= 1)
	{
		input[i] += input[i + stride];

		__syncthreads();
	}
	if (threadIdx.x == 0)
	{
		*output = input[0];
	}
}

__global__ void shared_sum_kernel(float *input, float *output)
{
	__shared__ float input_s[BLOCK_DIM];
	unsigned int     t = threadIdx.x;
	input_s[t]         = input[t] + input[t + BLOCK_DIM];
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		__syncthreads();
		if (threadIdx.x < stride)
		{
			input_s[t] += input_s[t + stride];
		}
	}
	if (threadIdx.x == 0)
	{
		*output = input_s[0];
	}
}

__global__ void segmented_sum_kernel(float *input, float *output)
{
	__shared__ float input_s[BLOCK_DIM];

	unsigned int t = threadIdx.x;
	unsigned int i = BLOCK_DIM * 2 * blockIdx.x + t;

	input_s[t] = input[i] + input[i + BLOCK_DIM];
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		__syncthreads();
		if (threadIdx.x < stride)
		{
			input_s[t] += input_s[t + stride];
		}
	}
	if (threadIdx.x == 0)
	{
		atomicAdd(output, input_s[0]);
	}
}
}        // namespace cuda_kernels::reduction