#pragma once

#include "matrix_ops.h"

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

template <typename T>
class TiledMatrixMultiplier : public MatrixMultiplier<T>
{
  public:
	TiledMatrixMultiplier(bool use_dynamic_shared_mem = false);
	void useDynamicSharedMemory(bool use_dynamic_shared_mem);
	void useThreadCoarsening(bool use_thread_coarsening);
	void multiply(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) override;

  private:
	void initDeviceProperties();

	bool use_dynamic_shared_mem_{false};
	bool use_thread_coarsening_{false};
	int dynamic_tile_width_ {16};
	size_t shared_mem_size_{2048};
};

extern template class TiledMatrixMultiplier<float>;