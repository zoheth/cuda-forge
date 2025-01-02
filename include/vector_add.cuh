#ifndef VECTOR_ADD_CUH
#define VECTOR_ADD_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t AddVectorsGpu(const float *a, const float *b, float *c, int size);

#ifdef __cplusplus
}
#endif

#endif // VECTOR_ADD_CUH