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
__global__ void matrixMulKernel(T *M, T *N, T *P, int M_height, int M_width /* = N_height*/, int N_width, int tile_width)
{
	extern __shared__ char shared_mem[];
	T                     *Mds = reinterpret_cast<T *>(shared_mem);
	T                     *Nds = reinterpret_cast<T *>(shared_mem) + tile_width * tile_width;

	int bx        = blockIdx.x;
	int by        = blockIdx.y;
	int tx        = threadIdx.x;
	int ty        = threadIdx.y;
	int Row       = by * tile_width + ty;
	int Col       = bx * tile_width + tx;
	T   Pvalue    = 0;
	int num_tiles = (M_width + tile_width - 1) / tile_width;

	for (int ph = 0; ph < num_tiles; ++ph)
	{
		if (Row < M_height && (ph * tile_width + tx) < M_width)
		{
			Mds[ty * tile_width + tx] = M[Row * M_width + ph * tile_width + tx];
		}
		else
		{
			Mds[ty * tile_width + tx] = 0;
		}
		if ((ph * tile_width + ty) < M_width && Col < N_width)
		{
			Nds[ty * tile_width + tx] = N[(ph * tile_width + ty) * N_width + Col];
		}
		else
		{
			Nds[ty * tile_width + tx] = 0;
		}

		__syncthreads();

		for (int k = 0; k < tile_width; ++k)
		{
			Pvalue += Mds[ty * tile_width + k] * Nds[k * tile_width + tx];
		}
		__syncthreads();
	}

	if (Row < M_height && Col < N_width)
	{
		P[Row * N_width + Col] = Pvalue;
	}
}

template <typename T>
TiledMatrixMultiplier<T>::TiledMatrixMultiplier(bool use_dynamic_shared_mem) :
    use_dynamic_shared_mem_(use_dynamic_shared_mem)
{
	if (use_dynamic_shared_mem_)
	{
		initDeviceProperties();
	}
}

template <typename T>
void TiledMatrixMultiplier<T>::multiply(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C)
{
	if (use_dynamic_shared_mem_)
	{
		dim3 threadsPerBlock(dynamic_tile_width_, dynamic_tile_width_);
		dim3 blocksPerGrid(
		    (B.width() + dynamic_tile_width_ - 1) / dynamic_tile_width_,
		    (A.height() + dynamic_tile_width_ - 1) / dynamic_tile_width_);

		matrixMulKernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size_>>>(
		    A.device_data(), B.device_data(), C.device_data(),
		    A.height(), A.width(), B.width(), dynamic_tile_width_);
	}
	else
	{
		dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
		dim3 blocksPerGrid(
		    (B.width() + TILE_WIDTH - 1) / TILE_WIDTH,
		    (A.height() + TILE_WIDTH - 1) / TILE_WIDTH);

		matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(
		    A.device_data(), B.device_data(), C.device_data(),
		    A.height(), A.width(), B.width());
	}

	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
template <typename T>
void TiledMatrixMultiplier<T>::initDeviceProperties()
{
	cudaDeviceProp prop;
	CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));

	size_t max_shared_mem        = prop.sharedMemPerBlock;
	int    max_threads_per_block = prop.maxThreadsPerBlock;

	// 需要的共享内存 = 2 * tile_width * tile_width * sizeof(T)
	// 需要的线程数 = tile_width * tile_width
	int max_tile_width          = static_cast<int>(sqrt(max_shared_mem / (2 * sizeof(T))));
	int thread_limit_tile_width = static_cast<int>(sqrt(max_threads_per_block));

	dynamic_tile_width_ = std::min(max_tile_width, thread_limit_tile_width);

	dynamic_tile_width_ = [](int x) {
		if (x == 0) return 1;
		int power = 1;
		while (power <= x) {
			power <<= 1;
		}
		return power >> 1;
	}(dynamic_tile_width_);

	shared_mem_size_ = 2 * dynamic_tile_width_ * dynamic_tile_width_ * sizeof(T);

	printf("Device: %s\n", prop.name);
	printf("Max shared memory per block: %zu\n", max_shared_mem);
	printf("Max shared memory per SM: %zu\n", prop.sharedMemPerMultiprocessor);
	printf("Max threads per block: %d\n", max_threads_per_block);
	printf("Selected tile width: %d\n", dynamic_tile_width_);
	printf("Required shared memory: %zu\n", shared_mem_size_);
}
