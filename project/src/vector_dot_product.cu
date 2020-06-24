#include <stdio.h>
#include <stdlib.h>
#include "util_header.h"
#include "vector_header.h"

#define BLOCK_SIZE 512

__global__
void createvectortosum(float *a, float *b, float* c, int N) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = gridDim.x * blockDim.x;

	for(int i = index; i < N; i+=stride) {
		c[i] = a[i]*b[i];
	}
}

float* execute_vectordotproduct(float* a, float* b, int N, size_t threadsForCreation, size_t blocksForCreation, int deviceId) {
	size_t size = N * sizeof(float);

	float *c = NULL;
	checkCudaErrorState(cudaMallocManaged(&c, size), "Couldn't alloc result vector for addition");

	checkCudaErrorState(cudaMemPrefetchAsync(c, size, deviceId), "Couldn't send for result vector to device");

	cudaStream_t vecCStream;
  	checkCudaErrorState(cudaStreamCreate(&vecCStream), "Couldn't create stream for result vector");
	
	providednumberinitwithstreams<<<threadsForCreation, blocksForCreation, 0, vecCStream>>>(c, N, 0);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector c with init values for addition");
	
	createvectortosum<<<threadsForCreation, blocksForCreation>>>(a, b, c, N);
	checkCudaErrorState(cudaGetLastError(), "Problem adding values into result");
	
	checkCudaErrorState(cudaDeviceSynchronize(), "Problem while completing device threads");

	size_t threads = BLOCK_SIZE;
	size_t blocks = N / threads + 1;
	for(int s = N; s >= 512; s= s / 512 + 1) {
		reduce<BLOCK_SIZE><<<threads, blocks>>>(c, s);
		blocks = blocks / threads + 1;	
		checkCudaErrorState(cudaDeviceSynchronize(), "Problem while completing device threads");
	}	

	cudaStreamDestroy(vecCStream);
	float final = 0;
	for(int i = 0; i < 512; ++i) {
		final += c[i];
	}

	cudaFree(c);

	check_vectordotproduct(a, b, final, N);

	float *toReturn = NULL;
	checkCudaErrorState(cudaMallocManaged(&toReturn, size), "Couldn't alloc result vector for addition");
	toReturn[0] = final;
	return toReturn;
}

void check_vectordotproduct(float *a, float *b, float result, int N) {
	float test = 0;
	for(int i = 0; i < N; ++i) {
		test += a[i] * b[i];
	}
	if(test - result > 1E-5) {
		printf("Vector dot product is incorrect\n");
		return;
	}
	printf("Test Passed\n");
}
