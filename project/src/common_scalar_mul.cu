#include <stdio.h>
#include <stdlib.h>
#include "common_header.h"
#include "util_header.h"

__global__
void scalarmultiplication(float *a, float k, float* result, int N) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = index; i < N; i += stride) {
		result[i] = a[i] * k;
	}
}

void check_scalarmultiplication(float *a, float k, float *c, int N) {
	for(int i = 0; i < N; i++) {
		if(a[i] * k - c[i] > 1E-5) {
			printf("Scalar multiplication result not correct\n");
			return;
		}
	}
	printf("Test Passed\n");
}

float* execute_scalar_multiplication(float *a, float k, int N, size_t threads, size_t blocks, int deviceId) {	
	size_t size = N * sizeof(float);

	float *c = NULL;
	checkCudaErrorState(cudaMallocManaged(&c, size), "Couldn't alloc result vector for addition");

	checkCudaErrorState(cudaMemPrefetchAsync(c, size, deviceId), "Couldn't send for result vector to device");

	cudaStream_t vecCStream;
  	checkCudaErrorState(cudaStreamCreate(&vecCStream), "Couldn't create stream for result vector");
	
	providednumberinitwithstreams<<<threads, blocks, 0, vecCStream>>>(c, N, 0);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector c with init values for addition");
	
	scalarmultiplication<<<threads, blocks>>>(a, k, c, N);
	checkCudaErrorState(cudaGetLastError(), "Problem adding values into result");
	
	checkCudaErrorState(cudaDeviceSynchronize(), "Problem while completing device threads");

	cudaStreamDestroy(vecCStream);

	check_scalarmultiplication(a, k, c, N);

	return c;
}


