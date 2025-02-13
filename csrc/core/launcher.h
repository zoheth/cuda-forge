#pragma once

#include "common.h"
#include "timer.h"

class KernelLauncher {
public:
	template <typename KernelFunc, typename... Args>
		static void launch(const std::string& kernel_tag,
						   KernelFunc kernel,
						   dim3 grid, dim3 block,
						   size_t shared_mem,
						   cudaStream_t stream,
						   Args&&... args)
	{
		validate_config(grid, block);

		CudaTimer timer(stream, kernel_tag);
		timer.start();

		kernel<<<grid, block, shared_mem, stream>>>(std::forward<Args>(args)...);

		float elapsed = timer.stop();
		if (elapsed > 0) {
			CudaTimer::add_record(kernel_tag, elapsed);
		}

		CUDA_CHECK(cudaGetLastError());
	}

	template <typename KernelFunc, typename... Args>
	static void launch(KernelFunc kernel, dim3 grid, dim3 block,
					   size_t shared_mem, cudaStream_t stream, Args&&... args) {
		validate_config(grid, block);
		kernel<<<grid, block, shared_mem, stream>>>(std::forward<Args>(args)...);
		CUDA_CHECK(cudaGetLastError());
	}

	static dim3 calc_grid_dim(size_t total_threads, dim3 block) {
		size_t blocks_x = (total_threads + block.x - 1) / block.x;
		return dim3(static_cast<unsigned>(blocks_x), 1, 1);
	}

private:
	static void validate_config(dim3 grid, dim3 block) {
		int max_threads_per_block = 0;
		CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_block,
										  cudaDevAttrMaxThreadsPerBlock, 0));
		if (block.x * block.y * block.z > max_threads_per_block) {
			throw std::runtime_error("Block size exceeds device limit");
		}
	}
};