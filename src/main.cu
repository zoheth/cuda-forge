#include "kernels/elementwise/test.h"

int main()
{
	ElementwiseBenchmark elementwise_benchmark(4096*1024);

	elementwise_benchmark.benchmark_f32();
	elementwise_benchmark.benchmark_f16();

	CudaTimer::print_all();

	return 0;
}