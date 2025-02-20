#pragma once

#include "core/timer.h"
#include "sum.cuh"

#include <thrust/device_vector.h>

class ReductionBenchmark
{
  public:
	ReductionBenchmark(int n) :
	    n_(n), d_input_(n), d_result_(1)
	{
		thrust::fill(d_input_.begin(), d_input_.end(), 1.0f);
		result_ = n;
	}

	void benchmark()
	{
		// for (int i=0; i<100; ++i)
		// {
		// 	thrust::fill(d_input_.begin(), d_input_.end(), 1.0f);
		// 	{
		// 		auto _ = KernelTimer("Sum Simple");
		// 		cuda_kernels::reduction::simple_sum_kernel<<<1, n_ / 2>>>(
		// 			thrust::raw_pointer_cast(d_input_.data()),
		// 			thrust::raw_pointer_cast(d_result_.data()));
		// 	}
		// }
		// verify();
		//
		// for (int i=0; i<100; ++i)
		// {
		// 	thrust::fill(d_input_.begin(), d_input_.end(), 1.0f);
		// 	{
		// 		auto _ = KernelTimer("Sum Simple 1");
		// 		cuda_kernels::reduction::simple_sum_kernel_1<<<1, n_ / 2>>>(
		// 			thrust::raw_pointer_cast(d_input_.data()),
		// 			thrust::raw_pointer_cast(d_result_.data()));
		// 	}
		// }
		// verify();
		//
		// for (int i=0; i<100; ++i)
		// {
		// 	thrust::fill(d_input_.begin(), d_input_.end(), 1.0f);
		// 	{
		// 		auto _ = KernelTimer("Sum Shared");
		// 		cuda_kernels::reduction::shared_sum_kernel<<<1, n_ / 2>>>(
		// 			thrust::raw_pointer_cast(d_input_.data()),
		// 			thrust::raw_pointer_cast(d_result_.data()));
		// 	}
		// }
		// verify();

		for (int i = 0; i < 100; ++i)
		{
			thrust::fill(d_input_.begin(), d_input_.end(), 1.0f);
			thrust::fill(d_result_.begin(), d_result_.end(), 0);
			{
				auto _ = KernelTimer("Sum");
				cuda_kernels::reduction::segmented_sum_kernel<<<n_ / block_size_ / 2, block_size_>>>(
				    thrust::raw_pointer_cast(d_input_.data()),
				    thrust::raw_pointer_cast(d_result_.data()));
			}
		}
		verify();
	}

	void verify()
	{
		if (result_ != d_result_[0])
		{
			std::cerr << "Error: " << result_ << " != " << d_result_[0] << std::endl;
		}
	}

  private:
	int n_{};

	int block_size_ = 1024;

	thrust::device_vector<float> d_input_;
	thrust::device_vector<float> d_result_;
	float                        result_{};
};