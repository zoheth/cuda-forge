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

	brent_kung_scan_block(XY, tx, 1);

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

__device__ void brent_kung_scan_block(float *data, unsigned int tx, unsigned raw_stride)
{
	for (unsigned int stride = 1; stride <= blockDim.x; stride <<= 1)
	{
		__syncthreads();
		unsigned int index = (tx + 1) * (stride * 2) - 1;
		if (index < 2 * blockDim.x)
		{
			data[index * raw_stride + raw_stride -1] += data[(index - stride) * raw_stride + raw_stride - 1];
		}
	}

	for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
	{
		__syncthreads();
		unsigned int index = (tx + 1) * (stride * 2) - 1;
		if (index + stride < 2 * blockDim.x)
		{
			data[(index + stride) * raw_stride + raw_stride -1] += data[index * raw_stride + raw_stride - 1];
		}
	}
	__syncthreads();
}

__device__ void scan_block(float *data, unsigned int c_factor)
{
	unsigned int tx = threadIdx.x;
	for (unsigned int i = 1; i < c_factor; ++i)
	{
		data[tx * c_factor + i] += data[tx * c_factor + i - 1];
		__syncthreads();
		data[(tx+blockDim.x) * c_factor + i] += data[(tx+blockDim.x) * c_factor + i - 1];
	}
	__syncthreads();

	brent_kung_scan_block(data, tx, c_factor);

	__syncthreads();

	if (tx > 0)
	{
		for (unsigned int i = 0; i < c_factor - 1; ++i)
		{
			data[tx * c_factor + i] += data[(tx - 1) * c_factor + c_factor - 1];
		}
	}
	__syncthreads();
	for (unsigned int i = 0; i < c_factor - 1; ++i)
	{
		data[(tx+blockDim.x) * c_factor + i] += data[((tx+blockDim.x) - 1) * c_factor + c_factor - 1];
	}
}

__global__ void coarsened_scan(float *X, float *Y, unsigned int N)
{
	__shared__ float XY[SECTION_SIZE * CFACTOR];
	unsigned int     start_i = blockIdx.x * blockDim.x * CFACTOR * 2 + threadIdx.x;

	for (unsigned int j = 0; j < CFACTOR * 2; ++j)
	{
		if (start_i + j * blockDim.x < N)
		{
			XY[threadIdx.x + j * blockDim.x] = X[start_i + j * blockDim.x];
		}
		else
		{
			XY[threadIdx.x + j * blockDim.x] = 0;
		}
	}

	__syncthreads();

	scan_block(XY, CFACTOR);

	__syncthreads();

	for (unsigned int j = 0; j < CFACTOR * 2; ++j)
	{
		if (start_i + j * blockDim.x < N)
		{
			Y[start_i + j * blockDim.x] = XY[threadIdx.x + j * blockDim.x];
		}
	}
}

__global__ void all_scan(float *X, float *Y, unsigned int N)
{
	scan_block(X, N/SECTION_SIZE);
	__syncthreads();
	for (unsigned int j = 0; j < N/SECTION_SIZE*2; ++j)
	{
		if (threadIdx.x + j * blockDim.x < N)
		{
			Y[threadIdx.x + j * blockDim.x] = X[threadIdx.x + j * blockDim.x];
		}
	}
}

__global__ void scan_phase1(float *X, float *Y, unsigned int N, float *block_sums)
{
	__shared__ float XY[SECTION_SIZE * CFACTOR];
	unsigned int    tx = threadIdx.x;
	unsigned int     start_i = blockIdx.x * blockDim.x * CFACTOR * 2 + tx;

	for (unsigned int j = 0; j < CFACTOR * 2; ++j)
	{
		if (start_i + j * blockDim.x < N)
		{
			XY[tx + j * blockDim.x] = X[start_i + j * blockDim.x];
		}
		else
		{
			XY[tx + j * blockDim.x] = 0;
		}
	}

	__syncthreads();

	scan_block(XY, CFACTOR);

	__syncthreads();

	for (unsigned int j = 0; j < CFACTOR * 2; ++j)
	{
		if (start_i + j * blockDim.x < N)
		{
			Y[start_i + j * blockDim.x] = XY[tx + j * blockDim.x];
		}
	}

	__syncthreads();
	if (tx == blockDim.x - 1)
	{
		block_sums[blockIdx.x] = XY[2 * blockDim.x * CFACTOR - 1];
	}

}

__global__ void scan_phase2(unsigned int N, float *block_sums)
{
	scan_block(block_sums, N / SECTION_SIZE);
}

__global__ void scan_phase3(float *Y, unsigned int N, float *block_sums)
{
	unsigned int i = blockIdx.x * blockDim.x * CFACTOR + threadIdx.x;
	if (blockIdx.x > 0)
	{
		for (int j = 0; j < CFACTOR; ++j)
		{
			if (i + j * blockDim.x < N)
			{
				Y[i + j * blockDim.x] += block_sums[blockIdx.x - 1];
			}
		}
	}
}
}        // namespace cuda_kernels::scan