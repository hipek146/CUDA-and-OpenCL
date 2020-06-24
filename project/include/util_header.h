#ifndef H_UTIL_HEADER_LS
#define H_UTIL_HEADER_LS
#include <cuda.h>
#include <curand_kernel.h>

__global__
void randomnumberinitwithstreams(curandState* globalState, float* figure, int n, float a, float b);

__global__
void providednumberinitwithstreams(float* figure, int n, float num);

void randomnumberinitwithoutstreams(float* figure, int n, float a, float b);

void providednumberinitwithoutstreams(float* figure, int n, float num);

__global__
void figurecopy(float* input, float* output, int n);

__global__
void setup_seeds(curandState* state, unsigned long seed, int n);

void checkCudaErrorState(cudaError_t error, const char* message);

void printvector(float *vector, int n);

void printmatrix(float *matrix, int n, int m);

#endif
