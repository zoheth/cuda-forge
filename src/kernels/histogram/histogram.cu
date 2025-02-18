#include "histogram.cuh"

namespace cuda_kernels::histogram
{
__global__ void histo_kernel(char *data, unsigned int length, unsigned int *histo)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length)
	{
		int alphabet_position = data[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26)
		{
			atomicAdd(&(histo[alphabet_position / 4]), 1);
		}
	}
}

__global__ void histo_private_kernel(char *data, unsigned int length, unsigned int *histo)
{
	__shared__ unsigned int histo_s[NUM_BINS];

	for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
	{
		histo_s[bin] = 0U;
	}

	__syncthreads();

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length)
	{
		int alphabet_position = data[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26)
		{
			atomicAdd(&(histo_s[alphabet_position / 4]), 1);
		}
	}

	__syncthreads();

	for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
	{
		unsigned int binValue = histo_s[bin];
		if (binValue > 0)
		{
			atomicAdd(&(histo[bin]), binValue);
		}
	}
}

__global__ void histo_private_c_kernel(char *data, unsigned int length, unsigned int *histo)
{
	__shared__ unsigned int histo_s[NUM_BINS];

	for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
	{
		histo_s[bin] = 0U;
	}

	__syncthreads();

	unsigned int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (unsigned int i = start_i; i< length; i+=blockDim.x*gridDim.x)
	{
		int alphabet_position = data[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26)
		{
			atomicAdd(&(histo_s[alphabet_position / 4]), 1);
		}
	}

	__syncthreads();

	for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
	{
		unsigned int binValue = histo_s[bin];
		if (binValue > 0)
		{
			atomicAdd(&(histo[bin]), binValue);
		}
	}
}
}        // namespace cuda_kernels::histogram