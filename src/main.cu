#include <cstdio>
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/std/span>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

template <int block_size>
__global__ void reduce(cuda::std::span<int const> data, cuda::std::span<int> result)
{
	using BlockReduce = cub::BlockReduce<int, block_size>;
	__shared__ typename BlockReduce::TempStorage temp_storage;

	int const index = threadIdx.x + blockIdx.x * blockDim.x;
	int       sum   = 0;
	if (index < data.size())
	{
		sum += data[index];
	}
	sum = BlockReduce(temp_storage).Sum(sum);

	if (threadIdx.x == 0)
	{
		cuda::atomic_ref<int, cuda::thread_scope_device> atomic_result(result.front());
		atomic_result.fetch_add(sum, cuda::memory_order_relaxed);
	}
}

int main()
{
	int const                  N = 1000;
	thrust::device_vector<int> data(N);
	thrust::fill(data.begin(), data.end(), 1);

	thrust::device_vector<int> kernel_result(1);

	constexpr int block_size = 256;
	int const     num_blocks = (N + block_size - 1) / block_size;
	reduce<block_size><<<num_blocks, block_size>>>(cuda::std::span<int const>(thrust::raw_pointer_cast(data.data()), data.size()),
	                                               cuda::std::span<int>(thrust::raw_pointer_cast(kernel_result.data()), 1));

	auto const err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cout << "Error:" << cudaGetErrorString(err) << std::endl;
		return -1;
	}

	int const custom_result = kernel_result.front();

	int const thrust_result = thrust::reduce(thrust::device, data.begin(), data.end());

	std::printf("Custom kernel sum: %d\n", custom_result);
	std::printf("Thrust reduce sum: %d\n", thrust_result);
	assert(kernel_result[0] == thrust_result);
	return 0;
}
