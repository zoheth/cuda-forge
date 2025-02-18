#pragma once

#include "histogram.cuh"

#include <thrust/device_vector.h>

#include <random>
#include <string>

inline std::string gen_random_string(size_t length)
{
	thread_local std::mt19937 gen(std::random_device{}());
	thread_local std::uniform_int_distribution<> dist('a', 'z');

	std::string result;
	result.reserve(length);

	for (int i=0; i<length; ++i)
	{
		result+= dist(gen);
	}
	return result;
}

class HistogramBenchmark
{
  public:
	HistogramBenchmark(uint32_t n) :
	    n_(n)
	{
		std::string str = gen_random_string(n);
		d_input_ = thrust::device_vector<char>(str.begin(), str.end());

		d_result_ = thrust::device_vector<uint32_t>(7);
		result_.resize(7);

		for (char c: str)
		{
			result_[(c-'a')/4]+=1;
		}

	}

	void benchmark()
	{
		int const num_blocks = (n_ + block_size_ - 1) / block_size_;
		for (int i = 0; i < 100; ++i)
		{
			thrust::fill(d_result_.begin(), d_result_.end(), 0);
			{
				auto _ = KernelTimer("Alphabet Histogram");
				cuda_kernels::histogram::histo_kernel<<<num_blocks, block_size_>>>(
					thrust::raw_pointer_cast(d_input_.data()),
					n_,
					thrust::raw_pointer_cast(d_result_.data()));
			}
		}
		verify();

		for (int i = 0; i < 100; ++i)
		{
			thrust::fill(d_result_.begin(), d_result_.end(), 0);
			{
				auto _ = KernelTimer("Alphabet Histogram Privatization");
				cuda_kernels::histogram::histo_private_kernel<<<num_blocks, block_size_>>>(
					thrust::raw_pointer_cast(d_input_.data()),
					n_,
					thrust::raw_pointer_cast(d_result_.data()));
			}
		}
		verify();

		for (int i = 0; i < 100; ++i)
		{
			thrust::fill(d_result_.begin(), d_result_.end(), 0);
			{
				auto _ = KernelTimer("Alphabet Histogram Privatization C");
				cuda_kernels::histogram::histo_private_c_kernel<<<(num_blocks+CFACTOR-1)/CFACTOR, block_size_>>>(
					thrust::raw_pointer_cast(d_input_.data()),
					n_,
					thrust::raw_pointer_cast(d_result_.data()));
			}
		}
		verify();
	}

  private:
	void verify()
	{
		for (int i=0; i< result_.size(); ++i)
		{
			if (result_[i]!=d_result_[i])
			{
				std::cout << "\033[31m❌: Mismatch at index " << i
					  << ": expected " << result_[i]
					  << ", but got " << d_result_[i]
					  << "\033[0m" << std::endl;
				break;
			}
		}
	}

	unsigned int n_;
	int block_size_ = 256;

	thrust::device_vector<char>         d_input_;
	thrust::device_vector<uint32_t> d_result_;

	std::vector<uint32_t> result_;

	std::mt19937 gen{std::random_device{}()};
};
