#pragma once

#include "matrix_ops.h"

#define TILE_WIDTH 16

template <typename T>
class TiledMatrixMultiplier : public MatrixMultiplier<T>
{
  public:
	void multiply(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) override;

  private:
	static constexpr int BLOCK_SIZE = 16;
};

extern template class TiledMatrixMultiplier<float>;