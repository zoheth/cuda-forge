#include "adds.cuh"

namespace cuda_kernels::elementwise
{

__global__ void add_f32_kernel(float *a, float *b, float *c, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) c[idx] = a[idx] + b[idx];
}
}        // namespace cuda_kernels::elementwise