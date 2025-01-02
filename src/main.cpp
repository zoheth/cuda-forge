#include <stdio.h>
#include <vector>
#include "vector_add.cuh"

int main()
{
	const int arraySize = 5000;

	std::vector<float> a(arraySize, 1.0f);
	std::vector<float> b(arraySize, 2.0f);
	std::vector<float> c(arraySize, 0.0f);

	cudaError_t cudaStatus = AddVectorsGpu(a.data(), b.data(), c.data(), arraySize);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addVectorsGPU failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	bool correct = true;
	for (int i = 0; i < arraySize; i++) {
		if (c[i] != 3.0f) {
			correct = false;
			break;
		}
	}

	printf("Vector addition %s\n", correct ? "PASSED" : "FAILED");

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}