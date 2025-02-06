#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(call)                                                 \
	{                                                                          \
		const cudaError_t error = call;                                        \
		if (error != cudaSuccess)                                              \
		{                                                                      \
			printf("CUDA Error: %s:%d, ", __FILE__, __LINE__);                 \
			printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
			exit(1);                                                           \
		}                                                                      \
	}

class CudaTimer
{
  public:
	CudaTimer()
	{
		CHECK_CUDA_ERROR(cudaEventCreate(&start_));
		CHECK_CUDA_ERROR(cudaEventCreate(&stop_));
	}

	~CudaTimer()
	{
		CHECK_CUDA_ERROR(cudaEventDestroy(start_));
		CHECK_CUDA_ERROR(cudaEventDestroy(stop_));
	}

	void start()
	{
		CHECK_CUDA_ERROR(cudaEventRecord(start_));
	}

	float stop()
	{
		float milliseconds = 0;
		CHECK_CUDA_ERROR(cudaEventRecord(stop_));
		CHECK_CUDA_ERROR(cudaEventSynchronize(stop_));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start_, stop_));
		return milliseconds;
	}

  private:
	cudaEvent_t start_, stop_;
};