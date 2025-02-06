#pragma once

#include "matrix_ops.h"

template <typename T>
class BasicMatrixMultiplier : public MatrixMultiplier<T>
{
  public:
	void multiply(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) override;

  private:
	static constexpr int BLOCK_SIZE = 16;
};

template <typename T>
__global__ void matrixMulKernel(T *A, T *B, T *C, int width)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < width && col < width)
	{
		T sum = 0;
		for (int i = 0; i < width; ++i)
		{
			sum += A[row * width + i] * B[i * width + col];
		}
		C[row * width + col] = sum;
	}
}

template <typename T>
void BasicMatrixMultiplier<T>::multiply(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C)
{
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid(
	    (B.width() + BLOCK_SIZE - 1) / BLOCK_SIZE,
	    (A.height() + BLOCK_SIZE - 1) / BLOCK_SIZE);

	matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(
	    A.device_data(), B.device_data(), C.device_data(), A.width());

	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}