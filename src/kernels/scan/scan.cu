#include "scan.cuh"

namespace cuda_kernels::scan
{

__global__ void kogge_stone_scan(float *X, float *Y, unsigned int N)
{
	__shared__ float XY[SECTION_SIZE];
	unsigned int     i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		XY[threadIdx.x] = X[i];
	}
	else
	{
		XY[threadIdx.x] = 0;
	}
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
	{
		__syncthreads();
		float temp;
		if (threadIdx.x >= stride)
		{
			temp = XY[threadIdx.x - stride] + XY[threadIdx.x];
		}
		__syncthreads();
		if (threadIdx.x >= stride)
		{
			XY[threadIdx.x] = temp;
		}
	}
	if (i < N)
	{
		Y[i] = XY[threadIdx.x];
	}
}

__global__ void brent_kung_scan(float *X, float *Y, unsigned int N)
{
	__shared__ float XY[SECTION_SIZE];
	unsigned int     i  = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int     tx = threadIdx.x;
	if (i < N)
	{
		XY[tx] = X[i];
	}
	else
	{
		XY[tx] = 0;
	}
	if (i + blockDim.x < N)
	{
		XY[tx + blockDim.x] = X[i + blockDim.x];
	}
	else
	{
		XY[tx + blockDim.x] = 0;
	}

	for (unsigned int stride = 1; stride <= blockDim.x; stride <<= 1)
	{
		__syncthreads();
		unsigned int index = (tx + 1) * (stride * 2) - 1;
		if (index < 2 * blockDim.x)
		{
			XY[index] += XY[index - stride];
		}
	}

	for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
	{
		__syncthreads();
		unsigned int index = (tx + 1) * (stride * 2) - 1;
		if (index + stride < 2 * blockDim.x)
		{
			XY[index + stride] += XY[index];
		}
	}

	__syncthreads();
	if (i < N)
	{
		Y[i] = XY[tx];
	}
	if (i + blockDim.x < N)
	{
		Y[i + blockDim.x] = XY[tx + blockDim.x];
	}
}

__device__ void scan_block(float *data, unsigned int size, unsigned int c_factor)
{
	
}

__global__ void scan_phase1(float *X, float *Y, unsigned int N, unsigned int *block_sums)
{
	__shared__ float XY[SECTION_SIZE];
	unsigned int     i  = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int     tx = threadIdx.x;
	if (i < N)
	{
		XY[tx] = X[i];
	}
	else
	{
		XY[tx] = 0;
	}
	if (i + blockDim.x < N)
	{
		XY[tx + blockDim.x] = X[i + blockDim.x];
	}
	else
	{
		XY[tx + blockDim.x] = 0;
	}

	for (unsigned int stride = 1; stride <= blockDim.x; stride <<= 1)
	{
		__syncthreads();
		unsigned int index = (tx + 1) * (stride * 2) - 1;
		if (index < 2 * blockDim.x)
		{
			XY[index] += XY[index - stride];
		}
	}

	for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
	{
		__syncthreads();
		unsigned int index = (tx + 1) * (stride * 2) - 1;
		if (index + stride < 2 * blockDim.x)
		{
			XY[index + stride] += XY[index];
		}
	}

	__syncthreads();
	if (i < N)
	{
		Y[i] = XY[tx];
	}

	__syncthreads();
	if (tx == blockDim.x - 1)
	{
		block_sums[blockIdx.x] = XY[2 * blockDim.x - 1];
	}

}

__global__ void scan_phase2(unsigned int N, unsigned int *block_sums)
{

}
}        // namespace cuda_kernels::scan