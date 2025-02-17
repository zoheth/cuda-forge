#include "core/context.h"
#include "core/launcher.h"
#include "core/memory.h"
#include "core/transfer.h"
#include "kernels/elementwise/adds.cuh"
#include <iostream>
#include <vector>

int main()
{
	int                n = 2048 * 2048;
	CudaMemory<float>  d_a(n);
	CudaMemory<float>  d_b(n);
	CudaMemory<float>  d_c(n);
	std::vector<float> h_a(n, 1.0f);

	DataTransfer::copy(d_a, h_a.data(), n);
	DataTransfer::copy(d_b, h_a.data(), n);

	DeviceContext ctx(0);
	dim3          block(256);
	dim3          grid = KernelLauncher::calc_grid_dim(n, block);

	{
		constexpr int num_warmup = 10;
		for (int i = 0; i < num_warmup; i++)
		{
			KernelLauncher::launch(
			    "Warmup",
			    cuda_kernels::elementwise::add_f32_kernel, grid, block, 0, nullptr,
			    d_a.data(), d_b.data(), d_c.data(), n);
		}

		cudaDeviceSynchronize();

		for (int i = 0; i < 100; i++)
		{
			KernelLauncher::launch(
			    "Vector Add F32",
			    cuda_kernels::elementwise::add_f32_kernel, grid, block, 0, nullptr,
			    d_a.data(), d_b.data(), d_c.data(), n);
		}
		cudaDeviceSynchronize();
	}

	grid = KernelLauncher::calc_grid_dim(n / 4, block);
	{
		constexpr int num_warmup = 10;
		for (int i = 0; i < num_warmup; i++)
		{
			KernelLauncher::launch(
			    "Warmup",
			    cuda_kernels::elementwise::add_f32x4_kernel, grid, block, 0, nullptr,
			    d_a.data(), d_b.data(), d_c.data(), n);
		}

		cudaDeviceSynchronize();

		for (int i = 0; i < 100; i++)
		{
			KernelLauncher::launch(
			    "Vector Add F32 x 4",
			    cuda_kernels::elementwise::add_f32x4_kernel, grid, block, 0, nullptr,
			    d_a.data(), d_b.data(), d_c.data(), n);
		}
		cudaDeviceSynchronize();
	}

	CudaTimer::print_all();

	std::vector<float> h_c(n);
	DataTransfer::copy(h_c.data(), d_c, n);
	std::cout << h_c[10] << std::endl;

	return 0;
}