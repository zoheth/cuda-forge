#pragma once

#include "common.h"

class DeviceContext
{
  public:
	explicit DeviceContext(int device_id)
	{
		CUDA_CHECK(cudaGetDevice(&original_device_));
		if (device_id != original_device_)
		{
			CUDA_CHECK(cudaSetDevice(device_id));
		}
	}

	~DeviceContext()
	{
		CUDA_CHECK_NOTHROW(cudaSetDevice(original_device_));
	}

  private:
	int original_device_;
};