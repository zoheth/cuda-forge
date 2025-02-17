#pragma once

#include "adds.cuh"
#include "benchmark.h"
#include "core/transfer.h"

#include <cuda/std/cmath>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <algorithm>

class ElementwiseBenchmark
{
  public:
	ElementwiseBenchmark(int n) :
	    n_(n), f32_a_(n), f32_b_(n), f32_c_(n), f32_c_ref_(n), f16_a_(n), f16_b_(n), f16_c_(n), f16_c_ref_(n)
	{
		init_data();
	}

	void benchmark_f32()
	{
		// Vector Add
		{
			auto _ = KernelTimer("F32 Add thrust");
			thrust::transform(f32_a_.begin(), f32_a_.end(), f32_b_.begin(), f32_c_ref_.begin(), thrust::plus<float>());
		}

		int const num_blocks = (n_ + block_size_ - 1) / block_size_;
		for (int i = 0; i < 100; ++i)
		{
			auto _ = KernelTimer("F32 Add");
			cuda_kernels::elementwise::add_f32_kernel<<<num_blocks, block_size_>>>(
			    MAKE_SPAN(f32_a_),
			    MAKE_SPAN(f32_b_),
			    MAKE_SPAN(f32_c_));
		}
		verify_32();

		for (int i = 0; i < 100; ++i)
		{
			auto _ = KernelTimer("F32 Add x4");
			cuda_kernels::elementwise::add_f32x4_kernel<<<num_blocks, block_size_ / 4>>>(
			    thrust::raw_pointer_cast(f32_a_.data()),
			    thrust::raw_pointer_cast(f32_b_.data()),
			    thrust::raw_pointer_cast(f32_c_.data()), n_);
		}
		verify_32();
	}

	void benchmark_f16()
	{
		{
			auto _ = KernelTimer("F16 Add thrust");
			thrust::transform(f16_a_.begin(), f16_a_.end(), f16_b_.begin(), f16_c_ref_.begin(), thrust::plus<float>());
		}

		int const num_blocks = (n_ + block_size_ - 1) / block_size_;
		for (int i = 0; i < 100; ++i)
		{
			auto _ = KernelTimer("F16 Add");
			cuda_kernels::elementwise::add_f16_kernel<<<num_blocks, block_size_>>>(
			    thrust::raw_pointer_cast(f16_a_.data()),
			    thrust::raw_pointer_cast(f16_b_.data()),
			    thrust::raw_pointer_cast(f16_c_.data()), n_);
		}

		verify_16();

		for (int i = 0; i < 100; ++i)
		{
			auto _ = KernelTimer("F16 Add x2");
			cuda_kernels::elementwise::add_f16x2_kernel<<<num_blocks, block_size_/2>>>(
			    thrust::raw_pointer_cast(f16_a_.data()),
			    thrust::raw_pointer_cast(f16_b_.data()),
			    thrust::raw_pointer_cast(f16_c_.data()), n_);
		}

		verify_16();

		for (int i = 0; i < 100; ++i)
		{
			auto _ = KernelTimer("F16 Add x8 pack");
			cuda_kernels::elementwise::add_f16x8_pack_kernel<<<num_blocks, block_size_/8>>>(
			    thrust::raw_pointer_cast(f16_a_.data()),
			    thrust::raw_pointer_cast(f16_b_.data()),
			    thrust::raw_pointer_cast(f16_c_.data()), n_);
		}

		verify_16();
	}

  private:
	void init_data()
	{
		thrust::fill(f32_a_.begin(), f32_a_.end(), 1.0f);
		thrust::fill(f32_b_.begin(), f32_b_.end(), 2.0f);

		thrust::fill(f16_a_.begin(), f16_a_.end(), 1.0);
		thrust::fill(f16_b_.begin(), f16_b_.end(), 2.0f);
	}

	void verify_32()
	{
		auto mismatch_pair = thrust::mismatch(f32_c_.begin(), f32_c_.end(), f32_c_ref_.begin());
		if (mismatch_pair.first != f32_c_.end())
		{
			size_t mismatch_index = mismatch_pair.first - f32_c_.begin();
			float  val1           = *mismatch_pair.first;
			float  val2           = *mismatch_pair.second;
			std::cout << "\033[31m❌: Mismatch at index " << mismatch_index
			          << ": expected " << val2
			          << ", but got " << val1
			          << "\033[0m" << std::endl;
		}
	}

	void verify_16()
	{
		auto mismatch_pair = thrust::mismatch(f16_c_.begin(), f16_c_.end(), f16_c_ref_.begin());
		if (mismatch_pair.first != f16_c_.end())
		{
			size_t mismatch_index = mismatch_pair.first - f16_c_.begin();
			half   val1_h         = (*mismatch_pair.first);
			half   val2_h         = (*mismatch_pair.second);
			std::cout << "\033[31m❌: Mismatch at index " << mismatch_index
			          << ": expected " << static_cast<float>(val2_h)
			          << ", but got " << static_cast<float>(val1_h)
			          << "\033[0m" << std::endl;
		}
	}

  private:
	int n_;

	int block_size_ = 1024;

	thrust::device_vector<float> f32_a_;
	thrust::device_vector<float> f32_b_;
	thrust::device_vector<float> f32_c_;
	thrust::device_vector<float> f32_c_ref_;

	thrust::device_vector<half> f16_a_;
	thrust::device_vector<half> f16_b_;
	thrust::device_vector<half> f16_c_;
	thrust::device_vector<half> f16_c_ref_;

	KernelBenchmark bench_;
};
