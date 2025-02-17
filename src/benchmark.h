#pragma once

#include "core/memory.h"
#include "core/launcher.h"

struct KernelShape {
	dim3 grid;
	dim3 block;
	size_t shared_mem;
};

class KernelBenchmark {
protected:
	int num_warmup_;
	int num_iters_;

public:
	KernelBenchmark(int warmup = 10, int iters = 100)
		: num_warmup_(warmup), num_iters_(iters) {}

	template<typename F, typename... Args>
	void run(const char* name, const KernelShape& shape, F kernel_func, Args&&... args) {
		// Warmup
		for(int i = 0; i < num_warmup_; i++) {
			KernelLauncher::launch("Warmup", kernel_func,
				shape.grid, shape.block, shape.shared_mem, nullptr,
				std::forward<Args>(args)...);
		}
		cudaDeviceSynchronize();

		for(int i = 0; i < num_iters_; i++) {
			KernelLauncher::launch(name, kernel_func,
				shape.grid, shape.block, shape.shared_mem, nullptr,
				std::forward<Args>(args)...);
		}
		cudaDeviceSynchronize();
	}
};