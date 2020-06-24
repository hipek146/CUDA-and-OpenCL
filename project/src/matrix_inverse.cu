#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "util_header.h"
#include "matrix_header.h"

float* execute_matrixinversion(float *A, int N, size_t threads, size_t blocks, int deviceId) {
	size_t size = N * N * sizeof(float);
	size_t sizePivot = N * sizeof(int);
	size_t sizeHv = N * sizeof(float);

	float *hv = NULL;
	checkCudaErrorState(cudaMallocManaged(&hv, sizeHv), "Couldn't alloc pivot vector");	

	checkCudaErrorState(cudaMemPrefetchAsync(hv, sizeHv, deviceId), "Couldn't send for result matrix to device");

	int *p = NULL;
	checkCudaErrorState(cudaMallocManaged(&p, sizePivot), "Couldn't alloc pivot vector");	

	checkCudaErrorState(cudaMemPrefetchAsync(p, sizePivot, deviceId), "Couldn't send for result matrix to device");

	createPivotForInversion<<<threads, blocks>>>(p, N);

	float *C = NULL;
	checkCudaErrorState(cudaMallocManaged(&C, size), "Couldn't alloc result matrix for inversion");

	checkCudaErrorState(cudaMemPrefetchAsync(C, size, deviceId), "Couldn't send for result matrix to device");

	figurecopy<<<threads, blocks>>>(A, C, N*N);
	
	checkCudaErrorState(cudaDeviceSynchronize(), "Problem while completing device threads");
	checkCudaErrorState(cudaGetLastError(), "Problem while copying matrix A to C for inveriosn");
	typedef std::chrono::high_resolution_clock Time;
	typedef std::chrono::milliseconds ms;
  	typedef std::chrono::duration<float> fsec;
	auto t0 = Time::now();
	float hr, vmax, tmp;
	int hi, r; 
	for (int i = 0; i < N; ++i) {
		vmax = fabs(C[i * N + i]);
		r = i;
		for (int j = i + 1; j < N; ++j) {
			tmp = fabs(C[j * N + i]);
			if (tmp > vmax) {
				vmax = tmp;
				r = j;
			}
		}
		if (fabs(vmax) < 1E-5) {
			return A; //% Singular matrix
		}
		if (r > i) {
			for(int ci = 0; ci < N; ci++) {
				tmp = C[i * N + ci];
				C[i * N + ci] = C[r * N + ci];
				C[r * N + ci] = tmp;
			}
			hi = p[i];
			p[i] = p[r];
			p[r] = hi;
		}
		hr = 1.0 / C[i * N + i];
		hrSolve<<<threads, blocks>>>(C, N, i, hr);
		solveLU<<<threads, blocks>>>(C, N, i, hr);
		checkCudaErrorState(cudaDeviceSynchronize(), "Problem while completing device threads");
	}
	for (int i = 0; i < N; ++i) {
		solveFinalPart1<<<threads, blocks>>>(C, N, i, p, hv);
		solveFinalPart2<<<threads, blocks>>>(C, N, i, hv);
	}
	auto t1 = Time::now();
	fsec fs = t1 - t0;
	printf("Czas cuda: %f\n", fs.count());
	
	checkCudaErrorState(cudaGetLastError(), "Problem creating final inverted matrix");
	
	checkCudaErrorState(cudaDeviceSynchronize(), "Problem while completing device threads");

  	cudaMemPrefetchAsync(C, size, deviceId);

	t0 = Time::now();
	check_matrixinversion(A, C, N);
	t1 = Time::now();
	fs = t1 - t0;
	printf("Czas cpu: %f\n", fs.count());

	return C;
}

__global__
void createPivotForInversion(int* p, int N) {
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < N; i += stride) {
		p[i] = i;
	}
}

__global__
void hrSolve(float *A, int N, int i, float hr) {
	unsigned int k = threadIdx.x + blockIdx.x * blockDim.x;

	if(k < N) {
		if (k == i) {
			A[k * N + i] = hr;
		} else {
			A[k * N + i] = A[k * N + i] * hr;
		}
	}
}

__global__
void solveLU(float *A, int N, int i, float hr) {
	unsigned int k = threadIdx.x + blockIdx.x * blockDim.x;


	if(k < N) {
		if (k != i) {
			for (int j = 0; j < N; ++j) {
				if (j != i) {
					A[j * N + k] = A[j * N + k] - A[j * N + i] * A[i * N + k];
				}
			}
			A[i * N + k] = -(hr * A[i * N + k]);
		}
	}
}

__global__
void solveFinalPart1(float *A, int N, int i, int* p, float *hv) {
	unsigned int k = threadIdx.x + blockIdx.x * blockDim.x;

	if (k < N) {
		hv[p[k]] = A[i * N + k];
	}
}

__global__
void solveFinalPart2(float *A, int N, int i, float *hv) {
	unsigned int k = threadIdx.x + blockIdx.x * blockDim.x;

	if (k < N) {
		A[i * N + k] = hv[k];
	}
}

void check_matrixinversion(float *A, float *IA, int n)
{

	int r, hi;
	float tmp, vmax, hr;
	int p[n];
	float hv[n];
	float* Ac;
	Ac = (float*) calloc(n*n, sizeof(float));
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < n; ++j) {
			Ac[i * n + j] = A[i * n + j];
		}
	}
	printf("Test\n");
	for (int i = 0; i < n; ++i) {
		p[i] = i;
	}
	for (int i = 0; i < n; ++i) {
		vmax = fabs(Ac[i * n + i]);
		r = i;
		for (int j = i + 1; j < n; ++j) {
			tmp = fabs(Ac[j * n + i]);
			if (tmp > vmax) {
				vmax = tmp;
				r = j;
			}
		}
		if (fabs(vmax) < 1E-5) {
			return; //% Singular matrix
		}
		if (r > i) {
			for(int ci = 0; ci < n; ci++) {
				tmp = Ac[i * n + ci];
				Ac[i * n + ci] = Ac[r * n + ci];
				Ac[r * n + ci] = tmp;
			}
			hi = p[i];
			p[i] = p[r];
			p[r] = hi;
		}
		hr = 1.0 / Ac[i * n + i];
		for (int k = 0; k < n; ++k) {
			Ac[k * n + i] = Ac[k * n + i] * hr;
		}
		Ac[i * n + i] = hr;
		for (int k = 0; k < n; ++k) {
			if (k != i) {
				for (int j = 0; j < n; ++j) {
					if (j != i) {
						tmp = Ac[j * n + k];
						Ac[j * n + k] = tmp - Ac[j * n + i] * Ac[i * n + k];
					}
				}
				tmp = Ac[i * n + k];
				Ac[i * n + k] = -(hr * tmp);
			}
		}
	}
	for (int i = 0; i < n; ++i) {
		for (int k = 0; k < n; ++k) {
			hv[p[k]] = Ac[i  * n + k];
		}
		for (int k = 0; k < n; ++k) {
			Ac[i * n + k] = hv[k];
		}
	}

	for (int i = 0; i < n*n; ++i) {
		if(!(fabs(Ac[i] - IA[i]) < 1E-4)) {
			printf("Matrix inversion result incorrect\n");
			return;
		}
	}
	printf("Test Passed\n");
	free(Ac);
}
