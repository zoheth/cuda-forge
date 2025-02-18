#include "kernels/elementwise/test.h"
#include "kernels/histogram/test.h"

int main()
{
	// ElementwiseBenchmark elementwise_benchmark(4096*1024);
	//
	// elementwise_benchmark.benchmark_f32();
	// elementwise_benchmark.benchmark_f16();

	HistogramBenchmark histogram_benchmark(1000000);
	histogram_benchmark.benchmark();

	CudaTimer::print_all();

	return 0;
}