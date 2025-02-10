#include "basic_matrix_mul.cuh"
#include "tiled_matrix_mul.cuh"
#include <iostream>

template <typename T>
bool verify_result(const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> &C)
{
	const T tolerance = 1e-3;

	// A: M_height x M_width
	// B: M_width x N_width
	// C: M_height x N_width
	const int M_height = A.height();
	const int M_width = A.width();    // =B.height()
	const int N_width = B.width();

	if (B.height() != M_width || C.height() != M_height || C.width() != N_width)
	{
		std::cout << "Matrix dimensions mismatch!" << std::endl;
		std::cout << "A: " << M_height << "x" << M_width << std::endl;
		std::cout << "B: " << B.height() << "x" << N_width << std::endl;
		std::cout << "C: " << C.height() << "x" << C.width() << std::endl;
		return false;
	}

	for (int i = 0; i < M_height; ++i)
	{
		for (int j = 0; j < N_width; ++j)
		{
			T sum = 0;
			for (int k = 0; k < M_width; ++k)
			{
				sum += A.host_data()[i * M_width + k] * B.host_data()[k * N_width + j];
			}
			if (std::abs(C.host_data()[i * N_width + j] - sum) > tolerance)
			{
				std::cout << "Verification failed at [" << i << "," << j << "]!" << std::endl;
				std::cout << "Expected: " << sum << std::endl;
				std::cout << "Got: " << C.host_data()[i * N_width + j] << std::endl;
				std::cout << "Difference: " << std::abs(C.host_data()[i * N_width + j] - sum) << std::endl;
				return false;
			}
		}
	}
	return true;
}

int main()
{
	constexpr int width  = 10000;
	constexpr int height = 8000;

	Matrix<float> A(4000, height);
	Matrix<float> B(width, 4000);
	Matrix<float> C(width, height);

	// Matrix<float> A(width, width);
	// Matrix<float> B(width, width);
	// Matrix<float> C(width, width);

	A.random_init();
	B.random_init();

	TiledMatrixMultiplier<float> multiplier;
	//BasicMatrixMultiplier<float> multiplier;

	A.to_device();
	B.to_device();

	CudaTimer timer;
	timer.start();

	multiplier.multiply(A, B, C);

	float ms = timer.stop();
	std::cout << "Kernel execution time: " << ms << " ms" << std::endl;

	C.to_host();
	// A.print();
	// B.print();
	// C.print();

	// if (verify_result(A, B, C))
	// {
	// 	std::cout << "Verification successful!" << std::endl;
	// }
}