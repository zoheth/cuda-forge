#pragma once

#include "scan.cuh"
#include <thrust/device_vector.h>

class ScanBenchmark
{
  public:
	ScanBenchmark(int n) :
	    n_(n), input_(n), result_(n), result_ref_(n)
	{
		unsigned int final_num_block = (n_ + block_size_ - 1) / block_size_;
		final_num_block              = (final_num_block + CFACTOR - 1) / CFACTOR;
		block_sums_                  = thrust::device_vector<float>(final_num_block);

		thrust::fill(input_.begin(), input_.end(), 1.0f);

		{
			auto _ = KernelTimer("Scan Thrust");
			thrust::inclusive_scan(input_.begin(), input_.end(), result_ref_.begin());
		}
	}

	void benchmark()
	{
		int const num_blocks = (n_ + block_size_ - 1) / block_size_;
		for (int i = 0; i < 100; ++i)
		{
			auto _ = KernelTimer("Scan Kogge-Stone");
			cuda_kernels::scan::kogge_stone_scan<<<num_blocks, block_size_>>>(
			    thrust::raw_pointer_cast(input_.data()),
			    thrust::raw_pointer_cast(result_.data()), n_);
		}
		verify();

		for (int i = 0; i < 100; ++i)
		{
			auto _ = KernelTimer("Scan Brent-Kung");
			cuda_kernels::scan::brent_kung_scan<<<num_blocks, block_size_ / 2>>>(
			    thrust::raw_pointer_cast(input_.data()),
			    thrust::raw_pointer_cast(result_.data()), n_);
		}
		verify();

		for (int i = 0; i < 100; ++i)
		{
			auto _ = KernelTimer("Scan Coarsened");
			cuda_kernels::scan::coarsened_scan<<<num_blocks / CFACTOR, block_size_ / 2>>>(
			    thrust::raw_pointer_cast(input_.data()),
			    thrust::raw_pointer_cast(result_.data()), n_);
		}
		verify();

		// for (int i = 0; i < 1; ++i)
		// {
		// 	auto _ = KernelTimer("Scan ALL");
		// 	cuda_kernels::scan::all_scan<<<1, block_size_/2>>>(
		// 	    thrust::raw_pointer_cast(input_.data()),
		// 	    thrust::raw_pointer_cast(result_.data()), n_);
		// }
		// verify();

		for (int i = 0; i < 100; ++i)
		{
			auto     _               = KernelTimer("Scan ALL");
			uint32_t final_num_block = (num_blocks + CFACTOR - 1) / CFACTOR;
			cuda_kernels::scan::scan_phase1<<<final_num_block, block_size_ / 2>>>(
			    thrust::raw_pointer_cast(input_.data()),
			    thrust::raw_pointer_cast(result_.data()),
			    n_,
			    thrust::raw_pointer_cast(block_sums_.data()));
			cuda_kernels::scan::scan_phase2<<<1, block_size_ / 2>>>(
			    final_num_block,
			    thrust::raw_pointer_cast(block_sums_.data()));
			cuda_kernels::scan::scan_phase3<<<final_num_block, block_size_>>>(thrust::raw_pointer_cast(result_.data()),
			                                                                  n_,
			                                                                  thrust::raw_pointer_cast(block_sums_.data()));
		}
		verify();
	}

	void verify()
	{
		auto mismatch_pair = thrust::mismatch(result_.begin(), result_.end(), result_ref_.begin());
		if (mismatch_pair.first != result_.end())
		{
			size_t mismatch_index = mismatch_pair.first - result_.begin();
			float  val1           = *mismatch_pair.first;
			float  val2           = *mismatch_pair.second;
			std::cout << "\033[31m❌: Mismatch at index " << mismatch_index
			          << ": expected " << val2
			          << ", but got " << val1
			          << "\033[0m" << std::endl;
		}
	}

  private:
	unsigned int n_;

	int block_size_ = 1024;

	thrust::device_vector<float> input_;
	thrust::device_vector<float> result_;

	thrust::device_vector<float> result_ref_;

	thrust::device_vector<float> block_sums_;
};