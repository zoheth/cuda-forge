#pragma once

#include "../include/matrix_ops.h"

template <typename T>
class BasicMatrixMultiplier : public MatrixMultiplier<T>
{
  public:
	void multiply(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) override;

  private:
	static constexpr int BLOCK_SIZE = 16;
};

template <typename T>
__global__ void matrixMulKernel(T *M, T *N, T *P, int M_height, int M_width, int N_width)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M_height && col < N_width)
	{
		T sum = 0;
		for (int i = 0; i < M_width; ++i)
		{
			sum += M[row * M_width + i] * N[i * N_width + col];
		}
		P[row * N_width + col] = sum;
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
	    A.device_data(), B.device_data(), C.device_data(), A.height(), A.width(), B.width());

	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}