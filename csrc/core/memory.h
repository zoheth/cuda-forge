#pragma once

#include "common.h"

template <typename T>
class CudaMemory
{
  public:
	explicit CudaMemory(size_t count = 0) :
	    count_(count), data_(nullptr)
	{
		if (count_ > 0)
			allocate();
	}

	CudaMemory(CudaMemory &&other) noexcept
	    :
	    count_(other.count_), data_(other.data_)
	{
		other.data_  = nullptr;
		other.count_ = 0;
	}

	CudaMemory &operator=(CudaMemory &&other) noexcept
	{
		if (this != &other)
		{
			release();
			data_        = other.data_;
			count_       = other.count_;
			other.data_  = nullptr;
			other.count_ = 0;
		}
		return *this;
	}

	CudaMemory(const CudaMemory &)            = delete;
	CudaMemory &operator=(const CudaMemory &) = delete;

	~CudaMemory()
	{
		release();
	}

	void allocate(size_t new_count)
	{
		if (data_)
			release();
		count_ = new_count;
		CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));
	}

	T *data() noexcept
	{
		return data_;
	}
	const T *data() const noexcept
	{
		return data_;
	}
	size_t size() const noexcept
	{
		return count_;
	}

	T &operator[](size_t idx)
	{
		return data_[idx];
	}
	const T &operator[](size_t idx) const
	{
		return data_[idx];
	}

  private:
	void allocate()
	{
		allocate(count_);
	}

	void release()
	{
		if (data_)
		{
			CUDA_CHECK_NOTHROW(cudaFree(data_));
			data_  = nullptr;
			count_ = 0;
		}
	}

	size_t count_;
	T     *data_;
};