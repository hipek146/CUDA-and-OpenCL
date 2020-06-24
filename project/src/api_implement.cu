#include <stdio.h>
#include <stdlib.h>
#include "api_header.h"
#include "util_header.h"
#include "common_header.h"
#include "vector_header.h"
#include "matrix_header.h"
#include <fstream>
#include <iostream>

void figure_type() {
	printf("Program started.\n");
	printf("Choose figure to execute functions on (v - vector, m - matrix, default vetor): ");
	char fType;
	scanf(" %c", &fType);
	switch(fType) {
		case 'v':
			vector_functions();
			break;
		case 'm':
			matrix_functions();
			break;
		default:
			vector_functions();
			break;
	}
}

void vector_functions() {
	printf("Vector functions.\n");
	printf("Choose function to execute for vector (a - adddition, s - subtraction, k - multiplication by scalar, d - dot product, default addition): ");
	char vFunc;
	scanf(" %c", &vFunc);
	switch(vFunc) {
		case 'a':
			vector_addition();
			break;
		case 's':
			vector_subtraction();
			break;
		case 'k':
			vector_scalar_multiplication();
			break;
		case 'd':
			vector_dot_product();
			break;
		default:
			vector_addition();
			break;
	}
}

void vector_addition() {
	printf("Vector addition.\n");
	printf("Choose a way to provide vectors values (p - provided number, r - random number, f - file [default random number): ");
	char vValue;
	scanf(" %c", &vValue);
	switch(vValue) {
		case 'p':
			vector_provided_number_addition();
			break;
		case 'r':
			vector_random_number_addition();
			break;
		case 'f':
			vector_file_function(execute_addition);
			break;
		default:
			vector_random_number_addition();
			break;
	}
}

void vector_random_number_addition() {
	printf("Vector random values.\n");
	printf("Provide number of elements: ");
	int N;
	scanf("%d", &N);
	printf("Provide range of random nambers (start end): ");
	float start, end;
	scanf("%f %f", &start, &end);
	
	size_t size = N * sizeof(float);
	float *a, *b;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of vector a");
	checkCudaErrorState(cudaMallocManaged(&b, size), "Problem while allocating memory of vector b");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for vector a to device");
	checkCudaErrorState(cudaMemPrefetchAsync(b, size, deviceId), "Couldn't send for vector b to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	cudaStream_t vecAStream, vecBStream;
	checkCudaErrorState(cudaStreamCreate(&vecAStream), "Couldn't create stream for vector a");
 	checkCudaErrorState(cudaStreamCreate(&vecBStream), "Couldn't create stream for vector b");

	curandState* state;
	cudaMalloc(&state, sizeof(curandState));
	setup_seeds<<<threads, blocks>>>(state, time(NULL), N);
	checkCudaErrorState(cudaGetLastError(), "Problem while setting seed for graphics random generator");
	
	randomnumberinitwithstreams<<<threads, blocks, 0, vecAStream>>>(state, a, N, start, end);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector a with random values");

	randomnumberinitwithstreams<<<threads, blocks, 0, vecBStream>>>(state, b, N, start, end);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector b with random values");

	float* c = execute_addition(a, b, N, threads, blocks, deviceId);
	//printvector(a, N);
	//printvector(b, N);
	//printvector(c, N);

	cudaStreamDestroy(vecAStream);
	cudaStreamDestroy(vecBStream);

	cudaFree(state);
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

void vector_provided_number_addition() {
	printf("Vector provided values.\n");
	printf("Provide number of elements: ");
	int N;
	scanf("%d", &N);
	printf("Provide value for vector a: ");
	float av;
	scanf("%f", &av);
	printf("Provide value for vector b: ");
	float bv;
	scanf("%f", &bv);

	size_t size = N * sizeof(float);
	float *a, *b;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of vector a");
	checkCudaErrorState(cudaMallocManaged(&b, size), "Problem while allocating memory of vector b");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for vector a to device");
	checkCudaErrorState(cudaMemPrefetchAsync(b, size, deviceId), "Couldn't send for vector b to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	cudaStream_t vecAStream, vecBStream;
	checkCudaErrorState(cudaStreamCreate(&vecAStream), "Couldn't create stream for vector a");
 	checkCudaErrorState(cudaStreamCreate(&vecBStream), "Couldn't create stream for vector b");

	providednumberinitwithstreams<<<threads, blocks, 0, vecAStream>>>(a, N, av);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector a with provided value");

	providednumberinitwithstreams<<<threads, blocks, 0, vecBStream>>>(b, N, bv);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector b with provided value");

	float* c = execute_addition(a, b, N, threads, blocks, deviceId);
	//printvector(a, N);
	//printvector(b, N);
	//printvector(c, N);

	cudaStreamDestroy(vecAStream);
	cudaStreamDestroy(vecBStream);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

void vector_subtraction() {
	printf("Vector subtraction.\n");
	printf("Choose a way to provide vectors values (p - provided number, r - random number, f - file [default random number): ");
	char vValue;
	scanf(" %c", &vValue);
	switch(vValue) {
		case 'p':
			vector_provided_number_subtraction();
			break;
		case 'r':
			vector_random_number_subtraction();
			break;
		case 'f':
			vector_file_function(execute_subtraction);
			break;
		default:
			vector_random_number_subtraction();
			break;
	}
}

void vector_random_number_subtraction() {
	printf("Vector random values.\n");
	printf("Provide number of elements: ");
	int N;
	scanf("%d", &N);
	printf("Provide range of random nambers (startt end): ");
	float start,end;
	scanf("%f %f", &start, &end);
	
	size_t size = N * sizeof(float);
	float *a, *b;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of vector a");
	checkCudaErrorState(cudaMallocManaged(&b, size), "Problem while allocating memory of vector b");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for vector a to device");
	checkCudaErrorState(cudaMemPrefetchAsync(b, size, deviceId), "Couldn't send for vector b to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	cudaStream_t vecAStream, vecBStream;
	checkCudaErrorState(cudaStreamCreate(&vecAStream), "Couldn't create stream for vector a");
 	checkCudaErrorState(cudaStreamCreate(&vecBStream), "Couldn't create stream for vector b");

	curandState* state;
	cudaMalloc(&state, sizeof(curandState));
	setup_seeds<<<threads, blocks>>>(state, time(NULL), N);
	checkCudaErrorState(cudaGetLastError(), "Problem while setting seed for graphics random generator");
	
	randomnumberinitwithstreams<<<threads, blocks, 0, vecAStream>>>(state, a, N, start, end);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector a with random values");

	randomnumberinitwithstreams<<<threads, blocks, 0, vecBStream>>>(state, b, N, start, end);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector b with random values");

	float* c = execute_subtraction(a, b, N, threads, blocks, deviceId);
	//printvector(a, N);
	//printvector(b, N);
	//printvector(c, N);

	cudaStreamDestroy(vecAStream);
	cudaStreamDestroy(vecBStream);

	cudaFree(state);
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

void vector_provided_number_subtraction() {
	printf("Vector provided values.\n");
	printf("Provide number of elements: ");
	int N;
	scanf("%d", &N);
	printf("Provide value for vector a: ");
	float av;
	scanf("%f", &av);
	printf("Provide value for vector b: ");
	float bv;
	scanf("%f", &bv);

	size_t size = N * sizeof(float);
	float *a, *b;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of vector a");
	checkCudaErrorState(cudaMallocManaged(&b, size), "Problem while allocating memory of vector b");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for vector a to device");
	checkCudaErrorState(cudaMemPrefetchAsync(b, size, deviceId), "Couldn't send for vector b to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	cudaStream_t vecAStream, vecBStream;
	checkCudaErrorState(cudaStreamCreate(&vecAStream), "Couldn't create stream for vector a");
 	checkCudaErrorState(cudaStreamCreate(&vecBStream), "Couldn't create stream for vector b");

	providednumberinitwithstreams<<<threads, blocks, 0, vecAStream>>>(a, N, av);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector a with provided value");

	providednumberinitwithstreams<<<threads, blocks, 0, vecBStream>>>(b, N, bv);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector b with provided value");

	float* c = execute_subtraction(a, b, N, threads, blocks, deviceId);
	//printvector(a, N);
	//printvector(b, N);
	//printvector(c, N);

	cudaStreamDestroy(vecAStream);
	cudaStreamDestroy(vecBStream);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

void vector_dot_product() {
	printf("Vector dot product.\n");
	printf("Choose a way to provide vectors values (p - provided number, r - random number, f - file [default random number): ");
	char vValue;
	scanf(" %c", &vValue);
	switch(vValue) {
		case 'p':
			vector_provided_number_dot_product();
			break;
		case 'r':
			vector_random_number_dot_product();
			break;
		case 'f':
			vector_file_function(execute_vectordotproduct);
			break;
		default:
			vector_random_number_dot_product();
			break;
	}
}

void vector_random_number_dot_product() {
	printf("Vector random values.\n");
	printf("Provide number of elements: ");
	int N;
	scanf("%d", &N);
	printf("Provide range of random nambers (start end): ");
	float start,end;
	scanf("%f %f", &start, &end);
	
	size_t size = N * sizeof(float);
	float *a, *b;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of vector a");
	checkCudaErrorState(cudaMallocManaged(&b, size), "Problem while allocating memory of vector b");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for vector a to device");
	checkCudaErrorState(cudaMemPrefetchAsync(b, size, deviceId), "Couldn't send for vector b to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	cudaStream_t vecAStream, vecBStream;
	checkCudaErrorState(cudaStreamCreate(&vecAStream), "Couldn't create stream for vector a");
 	checkCudaErrorState(cudaStreamCreate(&vecBStream), "Couldn't create stream for vector b");

	curandState* state;
	cudaMalloc(&state, sizeof(curandState));
	setup_seeds<<<threads, blocks>>>(state, time(NULL), N);
	checkCudaErrorState(cudaGetLastError(), "Problem while setting seed for graphics random generator");
	
	randomnumberinitwithstreams<<<threads, blocks, 0, vecAStream>>>(state, a, N, start, end);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector a with random values");

	randomnumberinitwithstreams<<<threads, blocks, 0, vecBStream>>>(state, b, N, start, end);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector b with random values");

	float* c = execute_vectordotproduct(a, b, N, threads, blocks, deviceId);
	//printvector(a, N);
	//printvector(b, N);
	//printf("Dot product of a and b: %f\n", c);

	cudaStreamDestroy(vecAStream);
	cudaStreamDestroy(vecBStream);

	cudaFree(state);
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

void vector_provided_number_dot_product() {
	printf("Vector provided values.\n");
	printf("Provide number of elements: ");
	int N;
	scanf("%d", &N);
	printf("Provide value for vector a: ");
	float av;
	scanf("%f", &av);
	printf("Provide value for vector b: ");
	float bv;
	scanf("%f", &bv);

	size_t size = N * sizeof(float);
	float *a, *b;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of vector a");
	checkCudaErrorState(cudaMallocManaged(&b, size), "Problem while allocating memory of vector b");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for vector a to device");
	checkCudaErrorState(cudaMemPrefetchAsync(b, size, deviceId), "Couldn't send for vector b to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	cudaStream_t vecAStream, vecBStream;
	checkCudaErrorState(cudaStreamCreate(&vecAStream), "Couldn't create stream for vector a");
 	checkCudaErrorState(cudaStreamCreate(&vecBStream), "Couldn't create stream for vector b");

	providednumberinitwithstreams<<<threads, blocks, 0, vecAStream>>>(a, N, av);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector a with provided value");

	providednumberinitwithstreams<<<threads, blocks, 0, vecBStream>>>(b, N, bv);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector b with provided value");

	float* c = execute_vectordotproduct(a, b, N, threads, blocks, deviceId);
	//printvector(a, N);
	//printvector(b, N);
	//printf("Dot product of a and b: %f\n",c); 

	cudaStreamDestroy(vecAStream);
	cudaStreamDestroy(vecBStream);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

void vector_scalar_multiplication() {
	printf("Vector multiplication by scalar.\n");
	printf("Choose a way to povide vectors values (p - provided number, r - random number, f - file [default random number): ");
	char vValue;
	scanf(" %c", &vValue);
	switch(vValue) {
		case 'p':
			vector_provided_number_scalar_multiplication();
			break;
		case 'r':
			vector_random_number_scalar_multiplication();
			break;
		case 'f':
			vector_file_number_scalar_multiplication();
			break;
		default:
			vector_random_number_scalar_multiplication();
			break;
	}
}

void vector_file_number_scalar_multiplication()
{
	printf("Vector file values.\n");
	printf("Provide file name: ");
	std::string name;
	std::cin >> name;
	printf("Provide value for scalar k: ");
	float kv;
	scanf("%f", &kv);
	std::ifstream file(name);
	int N;
	file >> N;
	float *a;
	size_t size = N * sizeof(float);
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of vector a");
	int f_a;
	int i = 0;
	while(i < N && file >> f_a)
	{
		a[i] = f_a;
		i++;
	}
	file.close();

	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for vector a to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	float* c = execute_scalar_multiplication(a, kv, N, threads, blocks, deviceId);

	cudaFree(a);
	cudaFree(c);
}

void vector_random_number_scalar_multiplication() {
	printf("Vector random values.\n");
	printf("Provide number of elements: ");
	int N;
	scanf("%d", &N);
	printf("Provide range of random nambers (start end): ");
	float start,end;
	scanf("%f %f", &start, &end);
	printf("Provide value for scalar k: ");
	float kv;
	scanf("%f", &kv);
	
	size_t size = N * sizeof(float);
	float *a;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of vector a");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for vector a to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	cudaStream_t vecAStream;
	checkCudaErrorState(cudaStreamCreate(&vecAStream), "Couldn't create stream for vector a");

	curandState* state;
	cudaMalloc(&state, sizeof(curandState));
	setup_seeds<<<threads, blocks>>>(state, time(NULL), N);
	checkCudaErrorState(cudaGetLastError(), "Problem while setting seed for graphics random generator");
	
	randomnumberinitwithstreams<<<threads, blocks, 0, vecAStream>>>(state, a, N, start, end);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector a with random values");

	float* c = execute_scalar_multiplication(a, kv, N, threads, blocks, deviceId);
	//printvector(a, N);
	//printvector(c, N);

	cudaStreamDestroy(vecAStream);

	cudaFree(state);
	cudaFree(a);
	cudaFree(c);
}

void vector_provided_number_scalar_multiplication() {
	printf("Vector provided values.\n");
	printf("Provide number of elements: ");
	int N;
	scanf("%d", &N);
	printf("Provide value for vector a: ");
	float av;
	scanf("%f", &av);
	printf("Provide value for scalar k: ");
	float kv;
	scanf("%f", &kv);

	size_t size = N * sizeof(float);
	float *a;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of vector a");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for vector a to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	cudaStream_t vecAStream;
	checkCudaErrorState(cudaStreamCreate(&vecAStream), "Couldn't create stream for vector a");

	providednumberinitwithstreams<<<threads, blocks, 0, vecAStream>>>(a, N, av);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing vector a with provided value");

	float* c = execute_scalar_multiplication(a, kv, N, threads, blocks, deviceId);
	//printvector(a, N);
	//printvector(c, N);

	cudaStreamDestroy(vecAStream);

	cudaFree(a);
	cudaFree(c);
}

void matrix_functions() {
	printf("Matrix functions.\n");
	printf("Choose function to execute for matrix (a - adddition, s - subtraction, k - multiplication by scalar, m - multiplication, i - inversion, default addition): ");
	char mFunc;
	scanf(" %c", &mFunc);
	switch(mFunc) {
		case 'a':
			matrix_addition();
			break;
		case 's':
			matrix_subtraction();
			break;
		case 'k':
			matrix_scalar_multiplication();
			break;
		case 'm':
			matrix_multiplication();
			break;
		case 'i':
			matrix_inversion();
			break;
		default:
			matrix_addition();
			break;
	}
}

void matrix_addition() {
	printf("Matrix addition.\n");
	printf("Choose a way to povide matrixes values (p - provided number, r - random number, f - file [default random number): ");
	char vValue;
	scanf(" %c", &vValue);
	switch(vValue) {
		case 'p':
			matrix_provided_number_addition();
			break;
		case 'r':
			matrix_random_number_addition();
			break;
		case 'f':
			matrix_file_function(execute_addition);
			break;
		default:
			matrix_random_number_addition();
			break;
	}
}

void matrix_random_number_addition() {
	printf("Matrix random values.\n");
	printf("Provide matrix dimentions: ");
	int N,M;
	scanf("%d %d", &N, &M);
	printf("Provide range of random nambers (start end): ");
	float start,end;
	scanf("%f %f", &start, &end);
	
	size_t size = N * M * sizeof(float);
	float *a, *b;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of matrix A");
	checkCudaErrorState(cudaMallocManaged(&b, size), "Problem while allocating memory of matrix B");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for matrix A to device");
	checkCudaErrorState(cudaMemPrefetchAsync(b, size, deviceId), "Couldn't send for matrix B to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	cudaStream_t vecAStream, vecBStream;
	checkCudaErrorState(cudaStreamCreate(&vecAStream), "Couldn't create stream for matrix A");
 	checkCudaErrorState(cudaStreamCreate(&vecBStream), "Couldn't create stream for matrix B");

	curandState* state;
	cudaMalloc(&state, sizeof(curandState));
	setup_seeds<<<threads, blocks>>>(state, time(NULL), N*M);
	checkCudaErrorState(cudaGetLastError(), "Problem while setting seed for graphics random generator");
	
	randomnumberinitwithstreams<<<threads, blocks, 0, vecAStream>>>(state, a, N*M, start, end);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing matrix A with random values");

	randomnumberinitwithstreams<<<threads, blocks, 0, vecBStream>>>(state, b, N*M, start, end);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing matrix B with random values");

	float* c = execute_addition(a, b, N*M, threads, blocks, deviceId);
	//printmatrix(a, N, M);
	//printmatrix(b, N, M);
	//printmatrix(c, N, M);

	cudaStreamDestroy(vecAStream);
	cudaStreamDestroy(vecBStream);

	cudaFree(state);
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

void matrix_provided_number_addition() {
	printf("Matrix provide value.\n");
	printf("Provide matrix dimentions: ");
	int N,M;
	scanf("%d %d", &N, &M);
	printf("Provide value for vector a: ");
	float av;
	scanf("%f", &av);
	printf("Provide value for vector b: ");
	float bv;
	scanf("%f", &bv);

	size_t size = N * M * sizeof(float);
	float *a, *b;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of matrix A");
	checkCudaErrorState(cudaMallocManaged(&b, size), "Problem while allocating memory of matrix B");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for matrix A to device");
	checkCudaErrorState(cudaMemPrefetchAsync(b, size, deviceId), "Couldn't send for matrix B to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	cudaStream_t vecAStream, vecBStream;
	checkCudaErrorState(cudaStreamCreate(&vecAStream), "Couldn't create stream for matrix A");
 	checkCudaErrorState(cudaStreamCreate(&vecBStream), "Couldn't create stream for matrix B");

	providednumberinitwithstreams<<<threads, blocks, 0, vecAStream>>>(a, N*M, av);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing matrix A with provided value");

	providednumberinitwithstreams<<<threads, blocks, 0, vecBStream>>>(b, N*M, bv);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing matrix B with provided value");

	float* c = execute_addition(a, b, N*M, threads, blocks, deviceId);
	//printmatrix(a, N, M);
	//printmatrix(b, N, M);
	//printmatrix(c, N, M);

	cudaStreamDestroy(vecAStream);
	cudaStreamDestroy(vecBStream);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

void matrix_subtraction() {
	printf("Matrix subtraction.\n");
	printf("Choose a way to povide matrixes values (p - provided number, r - random number, f - file [default random number): ");
	char vValue;
	scanf(" %c", &vValue);
	switch(vValue) {
		case 'p':
			matrix_provided_number_subtraction();
			break;
		case 'r':
			matrix_random_number_subtraction();
			break;
		case 'f':
			matrix_file_function(execute_subtraction);
			break;
		default:
			matrix_random_number_subtraction();
			break;
	}
}

void matrix_random_number_subtraction() {
	printf("Matrix random values.\n");
	printf("Provide matrix dimentions: ");
	int N,M;
	scanf("%d %d", &N, &M);
	printf("Provide range of random nambers (start end): ");
	float start,end;
	scanf("%f %f", &start, &end);
	
	size_t size = N * M * sizeof(float);
	float *a, *b;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of matrix A");
	checkCudaErrorState(cudaMallocManaged(&b, size), "Problem while allocating memory of matrix B");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for matrix A to device");
	checkCudaErrorState(cudaMemPrefetchAsync(b, size, deviceId), "Couldn't send for matrix B to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	cudaStream_t vecAStream, vecBStream;
	checkCudaErrorState(cudaStreamCreate(&vecAStream), "Couldn't create stream for matrix A");
 	checkCudaErrorState(cudaStreamCreate(&vecBStream), "Couldn't create stream for matrix B");

	curandState* state;
	cudaMalloc(&state, sizeof(curandState));
	setup_seeds<<<threads, blocks>>>(state, time(NULL), N*M);
	checkCudaErrorState(cudaGetLastError(), "Problem while setting seed for graphics random generator");
	
	randomnumberinitwithstreams<<<threads, blocks, 0, vecAStream>>>(state, a, N*M, start, end);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing matrix A with random values");

	randomnumberinitwithstreams<<<threads, blocks, 0, vecBStream>>>(state, b, N*M, start, end);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing matrix B with random values");

	float* c = execute_subtraction(a, b, N*M, threads, blocks, deviceId);
	//printmatrix(a, N, M);
	//printmatrix(b, N, M);
	//printmatrix(c, N, M);

	cudaStreamDestroy(vecAStream);
	cudaStreamDestroy(vecBStream);

	cudaFree(state);
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

void matrix_provided_number_subtraction() {
	printf("Matrix provide values.\n");
	printf("Provide matrix dimentions: ");
	int N,M;
	scanf("%d %d", &N, &M);
	printf("Provide value for vector a: ");
	float av;
	scanf("%f", &av);
	printf("Provide value for vector b: ");
	float bv;
	scanf("%f", &bv);

	size_t size = N * M * sizeof(float);
	float *a, *b;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of matrix A");
	checkCudaErrorState(cudaMallocManaged(&b, size), "Problem while allocating memory of matrix B");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for matrix A to device");
	checkCudaErrorState(cudaMemPrefetchAsync(b, size, deviceId), "Couldn't send for matrix B to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	cudaStream_t vecAStream, vecBStream;
	checkCudaErrorState(cudaStreamCreate(&vecAStream), "Couldn't create stream for matrix A");
 	checkCudaErrorState(cudaStreamCreate(&vecBStream), "Couldn't create stream for matrix B");

	providednumberinitwithstreams<<<threads, blocks, 0, vecAStream>>>(a, N*M, av);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing matrix A with provided value");

	providednumberinitwithstreams<<<threads, blocks, 0, vecBStream>>>(b, N*M, bv);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing matrix B with provided value");

	float* c = execute_addition(a, b, N*M, threads, blocks, deviceId);
	//printmatrix(a, N, M);
	//printmatrix(b, N, M);
	//printmatrix(c, N, M);

	cudaStreamDestroy(vecAStream);
	cudaStreamDestroy(vecBStream);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

void matrix_scalar_multiplication() {
	printf("Matrix multiplication by scalar.\n");
	printf("Choose a way to povide matrixess values (p - provided number, r - random number, f - file [default random number): ");
	char vValue;
	scanf(" %c", &vValue);
	switch(vValue) {
		case 'p':
			matrix_provided_number_scalar_multiplication();
			break;
		case 'r':
			matrix_random_number_scalar_multiplication();
			break;
		case 'f':
			matrix_file_number_scalar_multiplication();
			break;
		default:
			matrix_random_number_scalar_multiplication();
			break;
	}
}

void matrix_file_number_scalar_multiplication()
{
	printf("Matrix file values.\n");
	printf("Provide file name: ");
	std::string name;
	std::cin >> name;
	std::ifstream file(name);
	int N, M;
	file >> N >> M;

	printf("Provide value for scalar k: ");
	float kv;
	scanf("%f", &kv);

	size_t size = N * M * sizeof(float);
	float *a;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of matrix A");

	int i = 0;
	while(i < M)
	{
		int k = 0;
		while(k < N)
		{
			file >> a[i * N + k];
			k++;
		}
		i++;
	}
	file.close();

	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for matrix A to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	float* c = execute_scalar_multiplication(a, kv, N*M, threads, blocks, deviceId);
	//printmatrix(a, N, M);
	//printmatrix(b, N, M);
	//printmatrix(c, N, M);

	cudaFree(a);
	cudaFree(c);
}

void matrix_random_number_scalar_multiplication() {
	printf("Matrix random values.\n");
	printf("Provide matrix dimentions: ");
	int N,M;
	scanf("%d %d", &N, &M);
	printf("Provide range of random nambers (start end): ");
	float start,end;
	scanf("%f %f", &start, &end);
	printf("Provide value for scalar k: ");
	float kv;
	scanf("%f", &kv);
	
	size_t size = N * M * sizeof(float);
	float *a;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of matrix A");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for matrix A to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	cudaStream_t vecAStream;
	checkCudaErrorState(cudaStreamCreate(&vecAStream), "Couldn't create stream for matrix A");

	curandState* state;
	cudaMalloc(&state, sizeof(curandState));
	setup_seeds<<<threads, blocks>>>(state, time(NULL), N*M);
	checkCudaErrorState(cudaGetLastError(), "Problem while setting seed for graphics random generator");
	
	randomnumberinitwithstreams<<<threads, blocks, 0, vecAStream>>>(state, a, N*M, start, end);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing matrix A with random values");

	float* c = execute_scalar_multiplication(a, kv, N*M, threads, blocks, deviceId);
	//printmatrix(a, N, M);
	//printmatrix(c, N, M);

	cudaStreamDestroy(vecAStream);

	cudaFree(state);
	cudaFree(a);
	cudaFree(c);
}

void matrix_provided_number_scalar_multiplication() {
	printf("Matrix provide values.\n");
	printf("Provide matrix dimentions: ");
	int N,M;
	scanf("%d %d", &N, &M);
	printf("Provide value for vector a: ");
	float av;
	scanf("%f", &av);
	printf("Provide value for scalar k: ");
	float kv;
	scanf("%f", &kv);

	size_t size = N * M * sizeof(float);
	float *a;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of matrix A");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for matrix A to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	cudaStream_t vecAStream;
	checkCudaErrorState(cudaStreamCreate(&vecAStream), "Couldn't create stream for matrix A");

	providednumberinitwithstreams<<<threads, blocks, 0, vecAStream>>>(a, N*M, av);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing matrix A with provided value");

	float* c = execute_scalar_multiplication(a, kv, N*M, threads, blocks, deviceId);
	//printmatrix(a, N, M);
	//printmatrix(c, N, M);

	cudaStreamDestroy(vecAStream);

	cudaFree(a);
	cudaFree(c);
}

void matrix_multiplication() {
	printf("Matrix multiplication.\n");
	printf("Choose a way to povide vectors matrixess (p - provided number, r - random number, f - file [default random number): ");
	char vValue;
	scanf(" %c", &vValue);
	switch(vValue) {
		case 'p':
			matrix_provided_number_multiplication();
			break;
		case 'r':
			matrix_random_number_multiplication();
			break;
		case 'f':
			matrix_file_number_multiplication();
			break;
		default:
			matrix_random_number_multiplication();
			break;
	}
}

void matrix_file_number_multiplication()
{
	printf("Matrix file values.\n");
	printf("Provide file 1 name: ");
	std::string name;
	std::cin >> name;
	std::ifstream file(name);
	printf("Provide file 2 name: ");
	std::cin >> name;
	std::ifstream file2(name);
	int N, K, M;
	file >> N >> K;
	file2 >> K >> M;

	size_t sizeA = N * K * sizeof(float);
	size_t sizeB = K * M * sizeof(float);

	float *a, *b;
	checkCudaErrorState(cudaMallocManaged(&a, sizeA), "Problem while allocating memory of matrix A");
	checkCudaErrorState(cudaMallocManaged(&b, sizeB), "Problem while allocating memory of matrix B");

	int i = 0;
	while(i < K)
	{
		int j = 0;
		while(j < N)
		{
			file >> a[i * N + j];
			j++;
		}
		i++;
	}
	file.close();

	i = 0;
	while(i < M)
	{
		int j = 0;
		while(j < K)
		{
			file2 >> b[i * K + j];
			j++;
		}
		i++;
	}
	file2.close();

	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, sizeA, deviceId), "Couldn't send for matrix A to device");
	checkCudaErrorState(cudaMemPrefetchAsync(b, sizeB, deviceId), "Couldn't send for matrix B to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	float* c = execute_matrixmultiplication(a, b, N, K, M, threads, blocks, deviceId);
	//printmatrix(a, N, M);
	//printmatrix(b, N, M);
	//printmatrix(c, N, M);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

void matrix_random_number_multiplication() {
	printf("Matrix random values.\n");
	printf("Provide matrix dimentions (Ax AyBx By) (eg. 3 5 3): ");
	int N,K,M;
	scanf("%d %d %d", &N, &K, &M);
	printf("Provide range of random nambers (start end): ");
	float start,end;
	scanf("%f %f", &start, &end);
	
	size_t sizeA = N * K * sizeof(float);
	size_t sizeB = K * M * sizeof(float);
	float *a, *b;
	checkCudaErrorState(cudaMallocManaged(&a, sizeA), "Problem while allocating memory of matrix A");
	checkCudaErrorState(cudaMallocManaged(&b, sizeB), "Problem while allocating memory of matrix B");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, sizeA, deviceId), "Couldn't send for matrix A to device");
	checkCudaErrorState(cudaMemPrefetchAsync(b, sizeB, deviceId), "Couldn't send for matrix B to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;
	
	randomnumberinitwithoutstreams(a, N*K, start, end);

	randomnumberinitwithoutstreams(b, K*M, start, end);

	float* c = execute_matrixmultiplication(a, b, N, K, M, threads, blocks, deviceId);
	//printmatrix(a, N, K);
	//printmatrix(b, K, M);
	//printmatrix(c, N, M);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

void matrix_provided_number_multiplication() {
	printf("Matrix provided values.\n");
	printf("Provide matrix dimentions (Ax AyBx By) (eg. 3 5 3): ");
	int N,K,M;
	scanf("%d %d %d", &N, &K, &M);
	printf("Provide value for vector a: ");
	float av;
	scanf("%f", &av);
	printf("Provide value for vector b: ");
	float bv;
	scanf("%f", &bv);

	size_t sizeA = N * K * sizeof(float);
	size_t sizeB = K * M * sizeof(float);
	float *a, *b;
	checkCudaErrorState(cudaMallocManaged(&a, sizeA), "Problem while allocating memory of matrix A");
	checkCudaErrorState(cudaMallocManaged(&b, sizeB), "Problem while allocating memory of matrix B");

	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, sizeA, deviceId), "Couldn't send for matrix A to device");
	checkCudaErrorState(cudaMemPrefetchAsync(b, sizeB, deviceId), "Couldn't send for matrix B to device");

	providednumberinitwithoutstreams(a, N*K, av);

	providednumberinitwithoutstreams(b, K*M, bv);

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	float* c = execute_matrixmultiplication(a, b, N, K, M, threads, blocks, deviceId);
	//printmatrix(a, N, K);
	//printmatrix(b, K, M);
	//printmatrix(c, N, M);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

void matrix_inversion() {
	printf("Matrix inversion.\n");
	printf("Choose a way to povide matrixes values (r - random number, f - file [default random number): ");
	char vValue;
	scanf(" %c", &vValue);
	switch(vValue) {
		case 'r':
			matrix_random_number_inversion();
			break;
		case 'f':
			matrix_file_number_inversion();
			break;
		default:
			matrix_random_number_inversion();
			break;
	}
}
void matrix_file_number_inversion()
{
	printf("Matrix file values.\n");
	printf("Provide file name: ");
	std::string name;
	std::cin >> name;
	std::ifstream file(name);
	int N, M;
	file >> N >> M;

	size_t size = N * M * sizeof(float);
	float *a;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of matrix A");

	int i = 0;
	while(i < M)
	{
		int k = 0;
		while(k < N)
		{
			file >> a[i * N + k];
			k++;
		}
		i++;
	}
	file.close();

	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for matrix A to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	float* c = execute_matrixinversion(a, N, threads, blocks, deviceId);
	//printmatrix(a, N, M);
	//printmatrix(b, N, M);
	//printmatrix(c, N, M);

	cudaFree(a);
	cudaFree(c);
}
void matrix_random_number_inversion() {
	printf("Matrix random values only.\n");
	printf("Provide matrix dimention (for square matrix): ");
	int N;
	scanf("%d", &N);
	printf("Provide range of random nambers (start end): ");
	float start,end;
	scanf("%f %f", &start, &end);
	
	size_t size = N * N * sizeof(float);
	float *a;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of matrix A");
	
	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for matrix A to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	randomnumberinitwithoutstreams(a, N*N, start, end);
	checkCudaErrorState(cudaGetLastError(), "Problem while initializing matrix A with random values");

	float* c = execute_matrixinversion(a, N, threads, blocks, deviceId);
	//printf("Matrix A:\n");
	//printmatrix(a, N, N);
	//printf("Matrix C:\n");
	//printmatrix(c, N, N);

	cudaFree(a);
	cudaFree(c);
}

void vector_file_function(float* (*function)(float*, float*, int, size_t, size_t, int))
{
	printf("Vector file values.\n");
		printf("Provide file name: ");
		std::string name;
		std::cin >> name;
		std::ifstream file(name);
		int N;
		file >> N;
		float *a, *b;
		size_t size = N * sizeof(float);
		checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of vector a");
		checkCudaErrorState(cudaMallocManaged(&b, size), "Problem while allocating memory of vector b");
		int f_a, f_b;
		int i = 0;
		while(i < N && file >> f_a >> f_b)
		{
		    a[i] = f_a;
		    a[i] = f_b;
		    i++;
		}
		file.close();

		int deviceId;
		int numberOfSMs;

		cudaGetDevice(&deviceId);
		cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

		checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for vector a to device");
		checkCudaErrorState(cudaMemPrefetchAsync(b, size, deviceId), "Couldn't send for vector b to device");

		size_t threads = 256;
		size_t blocks = 32 * numberOfSMs;

		float* c = function(a, b, N, threads, blocks, deviceId);

		cudaFree(a);
		cudaFree(b);
		cudaFree(c);
}

void matrix_file_function(float* (*function)(float*, float*, int, size_t, size_t, int))
{
	printf("Matrix file values.\n");
	printf("Provide file 1 name: ");
	std::string name;
	std::cin >> name;
	std::ifstream file(name);
	printf("Provide file 2 name: ");
	std::cin >> name;
	std::ifstream file2(name);
	int N, M;
	file >> N >> M;

	size_t size = N * M * sizeof(float);
	float *a, *b;
	checkCudaErrorState(cudaMallocManaged(&a, size), "Problem while allocating memory of matrix A");
	checkCudaErrorState(cudaMallocManaged(&b, size), "Problem while allocating memory of matrix B");

	int i = 0;
	while(i < M)
	{
		int k = 0;
		while(k < N)
		{
			file >> a[i * N + k];
			k++;
		}
		i++;
	}
	file.close();

	i = 0;
	while(i < M)
	{
		int k = 0;
		while(k < N)
		{
			file2 >> b[i * N + k];
			k++;
		}
		i++;
	}
	file2.close();

	int deviceId;
	int numberOfSMs;

 	cudaGetDevice(&deviceId);
 	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	checkCudaErrorState(cudaMemPrefetchAsync(a, size, deviceId), "Couldn't send for matrix A to device");
	checkCudaErrorState(cudaMemPrefetchAsync(b, size, deviceId), "Couldn't send for matrix B to device");

	size_t threads = 256;
	size_t blocks = 32 * numberOfSMs;

	float* c = function(a, b, N*M, threads, blocks, deviceId);
	//printmatrix(a, N, M);
	//printmatrix(b, N, M);
	//printmatrix(c, N, M);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}
