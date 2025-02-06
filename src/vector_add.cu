#include "vector_add.cuh"

#include <memory>
#include <stdexcept>
#include <stdio.h>

template <typename T>
class CudaMemory
{
  public:
	explicit CudaMemory(size_t count) :
	    size_(count * sizeof(T))
	{
		const cudaError_t cuda_status = cudaMalloc(&data_, size_);
		if (cuda_status != cudaSuccess)
		{
			throw std::runtime_error("cudaMalloc failed");
		}
	}

	~CudaMemory()
	{
		if (data_)
		{
			cudaFree(data_);
		}
	}

	CudaMemory(const CudaMemory &)            = delete;
	CudaMemory &operator=(const CudaMemory &) = delete;

	bool CopyFromHost(const T *hostData)
	{
		return cudaMemcpy(data_, hostData, size_,
		                  cudaMemcpyHostToDevice) == cudaSuccess;
	}

	bool CopyToHost(T *hostData) const
	{
		return cudaMemcpy(hostData, data_, size_,
		                  cudaMemcpyDeviceToHost) == cudaSuccess;
	}

	T *get()
	{
		return data_;
	}
	const T *get() const
	{
		return data_;
	}

  private:
	T     *data_ = nullptr;
	size_t size_ = 0;
};

__global__ void addKernel(const float *a, const float *b, float *c, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		c[i] = a[i] + b[i];
	}
}

cudaError_t AddVectorsGpu(const float *a, const float *b, float *c, int size)
{
	try
	{
		CudaMemory<float> dev_a(size);
		CudaMemory<float> dev_b(size);
		CudaMemory<float> dev_c(size);

		if (!dev_a.CopyFromHost(a) || !dev_b.CopyFromHost(b))
		{
			return cudaErrorMemoryAllocation;
		}

		int block_size;
		int min_grid_size;
		cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
		                                   addKernel, 0, 0);

		const int num_blocks = (size + block_size - 1) / block_size;

		addKernel<<<num_blocks, block_size>>>(
		    dev_a.get(), dev_b.get(), dev_c.get(), size);

		cudaError_t kernel_status = cudaGetLastError();
		if (kernel_status != cudaSuccess)
		{
			return kernel_status;
		}

		if (cudaDeviceSynchronize() != cudaSuccess)
		{
			return cudaErrorLaunchFailure;
		}

		if (!dev_c.CopyToHost(c))
		{
			return cudaErrorUnknown;
		}

		return cudaSuccess;
	}
	catch (const std::exception &e)
	{
		fprintf(stderr, "CUDA error: %s\n", e.what());
		return cudaErrorUnknown;
	}
}