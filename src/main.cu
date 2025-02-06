#include "basic_matrix_mul.cuh"
#include <iostream>

template <typename T>
bool verify_result(const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> &C)
{
	const T   tolerance = 1e-4;
	const int width     = A.width();

	for (int i = 0; i < width; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			T sum = 0;
			for (int k = 0; k < width; ++k)
			{
				sum += A.host_data()[i * width + k] * B.host_data()[k * width + j];
			}
			if (std::abs(C.host_data()[i * width + j] - sum) > tolerance)
			{
				std::cout << C.host_data()[i * width + j] << std::endl;
				std::cout << sum << std::endl;
				std::cout << "Verification failed at [" << i << "," << j << "]!" << std::endl;
				return false;
			}
		}
	}
	return true;
}

int main()
{
	constexpr int width  = 1000;
	constexpr int height = 500;

	Matrix<float> A(400, height);
	Matrix<float> B(width, 400);
	Matrix<float> C(width, height);

	A.random_init();
	B.random_init();

	BasicMatrixMultiplier<float> multiplier;

	A.to_device();
	B.to_device();

	CudaTimer timer;
	timer.start();

	multiplier.multiply(A, B, C);

	float ms = timer.stop();
	std::cout << "Kernel execution time: " << ms << " ms" << std::endl;

	C.to_host();
	if (verify_result(A, B, C))
	{
		std::cout << "Verification successful!" << std::endl;
	}
}