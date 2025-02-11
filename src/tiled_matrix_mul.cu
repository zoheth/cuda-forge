#include "tiled_matrix_mul.cuh"

template class TiledMatrixMultiplier<float>;

template <typename T>
__global__ void matrixMulKernel(T *M, T *N, T *P, int M_height, int M_width /* = N_height*/, int N_width)
{
	__shared__ T Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ T Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	T Pvalue = 0;

	int num_tiles = (M_width + TILE_WIDTH - 1) / TILE_WIDTH;
	// 分块计算
	for (int ph = 0; ph < num_tiles; ++ph)
	{

		// 协作加载M和N的数据到共享内存
		if (Row < M_height && (ph * TILE_WIDTH + tx) < M_width)
		{
			Mds[ty][tx] = M[Row * M_width + ph * TILE_WIDTH + tx];
		}
		else
		{
			Mds[ty][tx] = 0;
		}

		if ((ph * TILE_WIDTH + ty) < M_width && Col < N_width)
		{
			Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * N_width + Col];
		}
		else
		{
			Nds[ty][tx] = 0;
		}
		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k)
		{
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}

	if (Row < M_height && Col < N_width)
	{
		P[Row * N_width + Col] = Pvalue;
	}
}

template <typename T>
void TiledMatrixMultiplier<T>::multiply(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C)
{
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 blocksPerGrid(
		(B.width() + TILE_WIDTH - 1) / TILE_WIDTH,
		(A.height() + TILE_WIDTH - 1) / TILE_WIDTH);

	matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(
		A.device_data(), B.device_data(), C.device_data(), A.height(), A.width(), B.width());

	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
