#pragma once

#include "memory.h"

class DataTransfer
{
  public:
	enum class Direction
	{
		HostToDevice,
		DeviceToHost,
		DeviceToDevice,
		HostToHost
	};

	template <typename T>
	static void copy(CudaMemory<T> &dst, const T *src, size_t count,
	                 Direction    dir    = Direction::HostToDevice,
	                 cudaStream_t stream = 0)
	{
		copy_impl(dst.data(), src, count * sizeof(T), dir, stream);
	}

	template <typename T>
	static void copy(T *dst, const CudaMemory<T> &src, size_t count,
	                 Direction    dir    = Direction::DeviceToHost,
	                 cudaStream_t stream = 0)
	{
		copy_impl(dst, src.data(), count * sizeof(T), dir, stream);
	}

  private:
	static void copy_impl(void *dst, const void *src, size_t bytes,
	                      Direction dir, cudaStream_t stream)
	{
		auto kind = [dir]() {
			switch (dir)
			{
				case Direction::HostToDevice:
					return cudaMemcpyHostToDevice;
				case Direction::DeviceToHost:
					return cudaMemcpyDeviceToHost;
				case Direction::DeviceToDevice:
					return cudaMemcpyDeviceToDevice;
				case Direction::HostToHost:
					return cudaMemcpyHostToHost;
				default:
					throw std::invalid_argument("Invalid copy direction");
			}
		}();

		if (stream)
		{
			CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, stream));
		}
		else
		{
			CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
		}
	}
};