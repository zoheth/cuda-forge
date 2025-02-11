#pragma once

#include "matrix_ops.h"

#define TILE_WIDTH 16

template <typename T>
class TiledMatrixMultiplier : public MatrixMultiplier<T>
{
  public:
	TiledMatrixMultiplier(bool use_dynamic_shared_mem = false);
	void multiply(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) override;

  private:
	void initDeviceProperties();

	bool use_dynamic_shared_mem_{false};
	int dynamic_tile_width_ {16};
	size_t shared_mem_size_{2048};
};

extern template class TiledMatrixMultiplier<float>;