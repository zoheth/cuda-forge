#pragma once

#include "cuda_utils.h"
#include <memory>

#include <vector>

template <typename T>
class Matrix
{
  public:
	Matrix(int width, int height) :
	    width_(width), height_(height), size_(width * height * sizeof(T))
	{
		h_data_ = std::make_unique<T[]>(width_ * height_);
		CHECK_CUDA_ERROR(cudaMalloc(&d_data_, size_));
	}

	~Matrix()
	{
		if (d_data_)
		{
			cudaFree(d_data_);
		}
	}

	Matrix(const Matrix &)            = delete;
	Matrix &operator=(const Matrix &) = delete;

	Matrix(Matrix &&)            = default;
	Matrix &operator=(Matrix &&) = default;

	T *device_data() const
	{
		return d_data_;
	}
	T *host_data() const
	{
		return h_data_.get();
	}

	void to_device()
	{
		CHECK_CUDA_ERROR(cudaMemcpy(d_data_, h_data_.get(), size_, cudaMemcpyHostToDevice));
	}

	void to_host()
	{
		CHECK_CUDA_ERROR(cudaMemcpy(h_data_.get(), d_data_, size_, cudaMemcpyDeviceToHost));
	}

	void random_init()
	{
		for (int i = 0; i < width_ * height_; ++i)
		{
			h_data_[i] = static_cast<T>(rand()) / RAND_MAX;
		}
	}

	int width() const
	{
		return width_;
	}
	int height() const
	{
		return height_;
	}
	size_t size() const
	{
		return size_;
	}

  private:
	int                  width_;
	int                  height_;
	size_t               size_;
	std::unique_ptr<T[]> h_data_;
	T                   *d_data_ = nullptr;
};

template <typename T>
class MatrixMultiplier
{
  public:
	virtual void multiply(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) = 0;
	virtual ~MatrixMultiplier()                                                 = default;
};