#pragma once
#include "adds.cuh"

namespace cuda_kernels::elementwise
{
struct LaunchParams
{
	dim3 grid;
	dim3 block;

	static LaunchParams get_1d_params(int N, int vec_size = 1)
	{
		LaunchParams params;
		params.grid = dim3(256 / vec_size);
		params.block = dim3((N+256-1)/256);
		return params;
	}
};

void add_f32(float*a, float*b, float*c, int N)
{
	auto params = LaunchParams::get_1d_params(N);
	add_f32_kernel<<<params.grid, params.block>>>(a, b, c, N);
}
}