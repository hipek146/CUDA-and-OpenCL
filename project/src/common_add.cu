#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "common_header.h"
#include "util_header.h"

__global__
void addition(float *a, float *b, float* c, int N) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = index; i < N; i += stride) {
		c[i] = a[i] + b[i];
	}
}

void check_addition(float *a, float *b, float *c, int N) {
	for(int i = 0; i < N; i++) {
		if(a[i] + b[i] - c[i] > 1E-5) {
			printf("Addition result not correct\n");
			return;
		}
	}
	printf("Test Passed\n");
}


float* execute_addition(float* a, float* b, int N, size_t threads, size_t blocks, int deviceId) {
	size_t size = N * sizeof(float);

	float *c = NULL;
	checkCudaErrorState(cudaMallocManaged(&c, size), "Couldn't alloc result vector for addition");

	checkCudaErrorState(cudaMemPrefetchAsync(c, size, deviceId), "Couldn't send for result vector to device");

	cudaStream_t vecCStream;
  	checkCudaErrorState(cudaStreamCreate(&vecCStream), "Couldn't create stream for result vector");
	
	providednumberinitwithstreams<<<threads, blocks, 0, vecCStream>>>(c, N, 0);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector c with init values for addition");
	
	addition<<<threads, blocks>>>(a, b, c, N);
	checkCudaErrorState(cudaGetLastError(), "Problem adding values into result");
	
	checkCudaErrorState(cudaDeviceSynchronize(), "Problem while completing device threads");

	cudaStreamDestroy(vecCStream);

	check_addition(a, b, c, N);

	return c;
}
