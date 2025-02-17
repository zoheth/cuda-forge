#include <torch/extension.h>
#include "kernels/elementwise/adds.cuh"

#include <cuda_runtime.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                  \
	if (((T).options().dtype() != (th_type)))                 \
	{                                                         \
		throw std::runtime_error("values must be " #th_type); \
	}

struct LaunchParams
{
	dim3 grid;
	dim3 block;
	static LaunchParams get_1d_params(int N, int vec_size = 1)
	{
		LaunchParams params;
		params.block = dim3(256 / vec_size);                // Threads per block
		params.grid = dim3((N + 256 - 1) / 256);           // Number of blocks
		return params;
	}
};

void add_f32(float*a, float*b, float*c, int N)
{
	auto params = LaunchParams::get_1d_params(N);
	cuda_kernels::elementwise::add_f32_kernel<<<params.grid, params.block>>>(a, b, c, N);
}

template<typename T>
void check_tensor(const torch::Tensor& tensor, torch::ScalarType dtype) {
	CHECK_TORCH_TENSOR_DTYPE(tensor, dtype);
	TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device");
	TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
}

void elementwise_add_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
	check_tensor<float>(a, torch::kFloat32);
	check_tensor<float>(b, torch::kFloat32);
	check_tensor<float>(c, torch::kFloat32);

	const int ndim = a.dim();
	int N;
	LaunchParams params;

	if (ndim != 2) {
		N = a.numel();
		params = LaunchParams::get_1d_params(N);
	} else {
		const int S = a.size(0);
		const int K = a.size(1);
		N = S * K;

		if (K <= 1024) {
			params.block = dim3(K);
			params.grid = dim3(S);
		} else {
			params = LaunchParams::get_1d_params(N);
		}
	}

	cuda_kernels::elementwise::add_f32_kernel<<<params.grid, params.block>>>(
		reinterpret_cast<float*>(a.data_ptr()),
		reinterpret_cast<float*>(b.data_ptr()),
		reinterpret_cast<float*>(c.data_ptr()),
		N
	);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("elementwise_add_f32", &elementwise_add_f32);
}