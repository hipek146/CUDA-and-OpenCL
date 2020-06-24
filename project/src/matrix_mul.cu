#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util_header.h"
#include "matrix_header.h"

float* execute_matrixmultiplication(float* A, float *B, int N, int K, int M, size_t threads, size_t blocks, int deviceId) {
	size_t size = N * M * sizeof(float);
	float *C = NULL;
	checkCudaErrorState(cudaMallocManaged(&C, size), "Couldn't alloc result vector for addition");

	checkCudaErrorState(cudaMemPrefetchAsync(C, size, deviceId), "Couldn't send for result vector to device");
	
	providednumberinitwithoutstreams(C, N*M, 0);

	matrixmultiplication<<<threads, blocks>>>(A, B, C, N, K, M);
	checkCudaErrorState(cudaGetLastError(), "Problem adding values into result");
	
	checkCudaErrorState(cudaDeviceSynchronize(), "Problem while completing device threads");

	cudaMemPrefetchAsync(C, size, deviceId);

	check_matrixmultiplication(A, B, C, N, K, M);

	return C;
}

__global__
void matrixmultiplication(float *A, float* B, float *C, int N, int K, int M) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = gridDim.x * blockDim.x;

	for(int i = index; i < N; i += stride) {
		for(int j = 0; j < M; ++j) {
			for(int k = 0; k < K; ++k) {
				//printf("%d %d %d %f %f ", i, j, k, A[i * N + k], B[k * K + j]);
				C[i * M + j] += A[i * K + k] * B[k * M + j];
			}
		}
	}
}

void check_matrixmultiplication(float *A, float* B, float *C, int N, int K, int M) {
	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < M; ++j) {
			float tmp = 0;
			for(int k = 0; k < K; ++k) {
				tmp += A[i * K + k] * B[k * M + j];
			}
			if (!(fabs(tmp - C[i * M + j]) < 1E-4)) {
				printf("Matrix multiplication result is incorrect\n");
				return;
			}
		}
	}
	printf("Test Passed\n");
}
