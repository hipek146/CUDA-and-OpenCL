#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util_header.h"

__global__
void setup_seeds(curandState* state, unsigned long seed, int n) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		curand_init(seed, idx, 0, &state[idx]);
	}
}

__global__
void randomnumberinitwithstreams(curandState* globalState, float* figure, int n, float a, float b) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i+=stride) {
		curandState localState = globalState[i];
		float random = curand_uniform(&localState);
		figure[i] = random * (b-a) + a;
		globalState[i] = localState;
	}
}

__global__
void providednumberinitwithstreams(float* figure, int n, float num) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i+=stride) {
		figure[i] = num;
	}
}

void randomnumberinitwithoutstreams(float* figure, int n, float a, float b) {
	for (int i = 0; i < n; ++i) {
		figure[i] = (float)rand() / RAND_MAX * (b-a) + a;
	} 
}

void providednumberinitwithoutstreams(float* figure, int n, float num) {
	for (int i = 0; i < n; ++i) {
		figure[i] = num;
	}
}

__global__
void figurecopy(float* input, float* output, int n) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i+= stride) {
		output[i] = input[i];
	}	
}

void checkCudaErrorState(cudaError_t error, const char* message) {
	if (error != CUDA_SUCCESS) {
		printf("%s, caues: %s\n", message, cudaGetErrorString(error));
	}
}

void printvector(float *vector, int n) {
	printf("Vector of size %d.\n", n);
	for ( int i = 0; i < n; ++i) {
		printf("%.1f ", vector[i]);
	}
	printf("\n");
}

void printmatrix(float *matrix, int n, int m) {
	printf("Matrix of size %dx%d.\n", n, m);
	for ( int i = 0; i < n; ++i) {
		for ( int j = 0; j < m; ++j) {
			printf("%.1f ", matrix[i * m + j]);
		}
		printf("\n");
	}
}
