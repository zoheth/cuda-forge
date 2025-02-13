#include "kernels/elementwise/adds_wrapper.cuh"

#include <iostream>
#include <vector>

int main()
{
	std::vector<float> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

	float *d_a = nullptr;
	float *d_b = nullptr;
	CUDA_CHECK(cudaMemcpy(d_a, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));


}