#include "kernels/elementwise/test.h"
#include "kernels/histogram/test.h"
#include "kernels/reduction/test.h"
#include "kernels/scan/test.h"

int main()
{
	// ElementwiseBenchmark elementwise_benchmark(4096*1024);
	//
	// elementwise_benchmark.benchmark_f32();
	// elementwise_benchmark.benchmark_f16();

	// HistogramBenchmark histogram_benchmark(1000000);
	// histogram_benchmark.benchmark();

	// ReductionBenchmark reduction_benchmark(4096*4096);
	// reduction_benchmark.benchmark();

	ScanBenchmark scan_benchmark(4096*1024);
	scan_benchmark.benchmark();

	CudaTimer::print_all();

	return 0;
}