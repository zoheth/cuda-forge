#pragma once

#include "core/common.h"

#define NUM_BINS 7

namespace cuda_kernels::histogram
{
__global__ void histo_kernel(char *data, unsigned int length, unsigned int *histo);

__global__ void histo_private_kernel(char *data, unsigned int length, unsigned int *histo);

__global__ void histo_private_c_kernel(char *data, unsigned int length, unsigned int *histo);
}