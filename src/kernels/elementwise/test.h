#pragma once

#include "benchmark.h"
#include "core/transfer.h"
#include "adds.cuh"

#include <algorithm>

class ElementwiseBenchmark
{
public:
	ElementwiseBenchmark(int n) : n_(n), f32_data_(n), f16_data_(n), bench_() {
		init_data();
	}

	void benchmark_f32() {
		// Vector Add
		{
			KernelShape shape{
				KernelLauncher::calc_grid_dim(n_, 256),
				dim3(256),
				0
			};
			bench_.run("F32 Add", shape,
				cuda_kernels::elementwise::add_f32_kernel,
				f32_data_.d_a.data(), f32_data_.d_b.data(),
				f32_data_.d_c.data(), n_);
		}

		// Vector Add x4
		{
			KernelShape shape{
				KernelLauncher::calc_grid_dim(n_/4, 256),
				dim3(256),
				0
			};
			bench_.run("F32 Add x4", shape,
				cuda_kernels::elementwise::add_f32x4_kernel,
				f32_data_.d_a.data(), f32_data_.d_b.data(),
				f32_data_.d_c.data(), n_);
		}
	}

	void benchmark_f16() {
		// Similar implementation for F16 kernels
	}

private:
	void init_data()
	{
		std::fill(f32_data_.h_ref.begin(), f32_data_.h_ref.end(), 1.0f);
		DataTransfer::copy(f32_data_.d_a, f32_data_.h_ref.data(), n_);
		DataTransfer::copy(f32_data_.d_b, f32_data_.h_ref.data(), n_);

		std::transform(f32_data_.h_ref.begin(), f32_data_.h_ref.end(), f16_data_.h_ref.begin(), __float2half);
		DataTransfer::copy(f16_data_.d_a, f16_data_.h_ref.data(), n_);
		DataTransfer::copy(f16_data_.d_b, f16_data_.h_ref.data(), n_);
	}

private:
	int n_;

	template<typename T>
	struct TestData {
		CudaMemory<T> d_a, d_b, d_c;
		std::vector<T> h_ref;

		TestData(int n) : d_a(n), d_b(n), d_c(n), h_ref(n) {}
	};

	TestData<float> f32_data_;
	TestData<half> f16_data_;
	KernelBenchmark bench_;
};
