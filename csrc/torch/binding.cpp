#include <torch/extension.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                  \
	if (((T).options().dtype() != (th_type)))                 \
	{                                                         \
		throw std::runtime_error("values must be " #th_type); \
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

	int N = a.numel();
	cuda_kernels::elementwise::add_f32(
		reinterpret_cast<float*>(a.data_ptr()),
		reinterpret_cast<float*>(b.data_ptr()),
		reinterpret_cast<float*>(c.data_ptr()),
		N
	);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("elementwise_add_f32", &elementwise_add_f32);
}