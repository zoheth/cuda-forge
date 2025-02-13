#include "core/context.h"
#include "core/launcher.h"
#include "core/memory.h"
#include "core/transfer.h"
#include "kernels/elementwise/adds.cuh"

#include <iostream>
#include <vector>

int main()
{
	int n = 1024;
	CudaMemory<float> d_a(n);
	CudaMemory<float> d_b(n);
	CudaMemory<float> d_c(n);
	std::vector<float> h_a(n, 1.0f);

	DataTransfer::copy(d_a, h_a.data(), n);
	DataTransfer::copy(d_b, h_a.data(), n);

	DeviceContext ctx(0);

	dim3 block(256);
	dim3 grid = KernelLauncher::calc_grid_dim(n, block);
	KernelLauncher::launch(
		"Vector Add",
		cuda_kernels::elementwise::add_f32_kernel, grid, block, 0, nullptr,
		d_a.data(), d_b.data(), d_c.data(), n
		);

	CudaTimer::print_all();

	std::vector<float> h_c(n);
	DataTransfer::copy(h_c.data(), d_c, n);

	std::cout<<h_c[10]<<std::endl;
}