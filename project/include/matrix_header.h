#ifndef H_MATRIX_HEADER_LS
#define H_MATRIX_HEADER_LS

float* execute_matrixmultiplication(float* A, float *B, int N, int K, int M, size_t threads, size_t blocks, int deviceId);

float* execute_matrixinversion(float *A, int N, size_t threads, size_t blocks, int deviceId);

void check_matrixinversion(float *A, float *IA, int N);

void check_matrixmultiplication(float *A, float *B, float *C, int N, int K, int M);

template<int blockAx, int blockABm, int blockBy>
__global__
void matrixmultiplicationforsquare(float* A, float* B, float* C, int N, int K, int M) {
	__shared__ float partA[blockAx * blockABm];
	__shared__ float partB[blockABm * blockBy];
	__shared__ float partC[blockAx * blockBy];

/*	for(int i = 0; i < blockABm; ++i) {
		for(int j = 0; j < blockAx; ++j) {
			partA[j * blockAx + i] = A[startA];
		}
	}*/
}

__global__
void matrixmultiplication(float *A, float* B, float *C, int N, int K, int M);

__global__
void createPivotForInversion(int* p, int N);
__global__
void hrSolve(float *A, int N, int i, float hr);
__global__
void solveLU(float *A, int N, int i, float hr);
__global__
void solveFinalPart2(float *A, int N, int i, float *hv);
__global__
void solveFinalPart1(float *A, int N, int i, int *p, float *hv);

#endif
