/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
matrixAdd1D1D(const float *MatA, const float *MatB, float *MatC, int nx, int ny)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < nx * ny)
    {
    	MatC[i] = MatA[i] + MatB[i];
    }
}
__global__ void
matrixHadamard1D1D(const float *MatA, const float *MatB, float *MatC, int nx, int ny)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < nx * ny)
    {
    	MatC[i] = MatA[i] * MatB[i];
    }
}

__global__ void
matrixAdd2D2D(const float *MatA, const float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int idx = iy*nx + ix;

    if (ix < nx && iy < ny)
    {
    	MatC[idx] = MatA[idx] + MatB[idx];
    }
}

__global__ void
matrixHadamard2D2D(const float *MatA, const float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int idx = iy*nx + ix;

    if (ix < nx && iy < ny)
    {
    	MatC[idx] = MatA[idx] * MatB[idx];
    }
}

void initilization(float **h_MatA, float **h_MatB, float **h_MatC,
		float **d_MatA, float **d_MatB, float **d_MatC,
		size_t size, int nx, int ny, cudaError_t err) {

	*h_MatA = (float *)malloc(size);

	// Allocate the host input vector B
	*h_MatB = (float *)malloc(size);

	// Allocate the host output vector C
	*h_MatC = (float *)malloc(size);

	// Verify that allocations succeeded
	if (*h_MatA == NULL || *h_MatB == NULL || *h_MatC == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < nx; ++i)
	{
		for (int j = 0; j < ny; ++j) {
			(*h_MatA)[j*nx+i] = rand()/(float)RAND_MAX;
			(*h_MatB)[j*nx+i] = rand()/(float)RAND_MAX;
		}
	}

	// Allocate the device input vector A
	err = cudaMalloc((void **)d_MatA, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	err = cudaMalloc((void **)d_MatB, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate Matrix vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	err = cudaMalloc((void **)d_MatC, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate Matrix vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(*d_MatA, *h_MatA, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy Matrix A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(*d_MatB, *h_MatB, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy Matrix B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void checkMatAdd(float* h_MatA, float* h_MatB, float* h_MatC,
		float* d_MatC, int nx, int ny, size_t size, cudaError_t err) {
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_MatC, d_MatC, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy Matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Verify that the result vector is correct
	for (int i = 0; i < nx; ++i)
	{
		for (int j = 0; j < ny; ++j) {
			if (fabs(h_MatA[j*nx+i] + h_MatB[j*nx+i] - h_MatC[j*nx+i]) > 1e-5)
			{
				fprintf(stderr, "Result verification failed at element %d x %d!\n", i, j);
				exit(EXIT_FAILURE);
			}
		}
	}

	printf("Test PASSED\n");
}

void checkHadamardAdd(float* h_MatA, float* h_MatB, float* h_MatC,
		float* d_MatC, int nx, int ny, size_t size, cudaError_t err) {
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_MatC, d_MatC, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy Matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Verify that the result vector is correct
	for (int i = 0; i < nx; ++i)
	{
		for (int j = 0; j < ny; ++j) {
			if (fabs(h_MatA[j*nx+i] * h_MatB[j*nx+i] - h_MatC[j*nx+i]) > 1e-5)
			{
				fprintf(stderr, "Result verification failed at element %d x %d!\n", i, j);
				exit(EXIT_FAILURE);
			}
		}
	}

	printf("Test PASSED\n");
}

void execute2D2D(float* d_MatA, float* d_MatB, float* d_MatC, int nx, int ny, cudaError_t err) {
    // Launch the Vector Add CUDA Kernel
    dim3 threads(nx/16, ny/16);
    dim3 blocks(threads.x*16 + 1, threads.y*16 + 1);
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocks.x*blocks.x, threads.y*threads.y);
    //matrixAdd2D2D<<<blocks, threads>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    matrixHadamard2D2D<<<blocks, threads>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "matrixAdd2D2D (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void execute1D1D(float* d_MatA, float* d_MatB, float* d_MatC, int nx, int ny, cudaError_t err) {
    // Launch the Vector Add CUDA Kernel
    int threads = (nx/16)*(ny/16);
    int blocks = (nx + 1)*(nx + 1);
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocks, threads);
    matrixAdd1D1D<<<blocks, threads>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    //matrixHadamard1D1D<<<blocks, threads>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "matrixAdd2D2D (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void cleanUp(float* h_MatA, float* h_MatB, float* h_MatC,
		float* d_MatA, float* d_MatB, float* d_MatC, cudaError_t err) {

    // Free device global memory
    err = cudaFree(d_MatA);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_MatB);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_MatC);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_MatA);
    free(h_MatB);
    free(h_MatC);

    printf("Done\n");
}
/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int nx = 100;
    int ny = 100;
	size_t size = nx * ny * sizeof(float);
    printf("[Matrix calculation of %dx%d elements]\n", nx, ny);

    // Allocate the host input vector A
    float *h_MatA = NULL;

    // Allocate the host input vector B
    float *h_MatB = NULL;

    // Allocate the host output vector C
    float *h_MatC = NULL;

    // Allocate the device input vector A
    float *d_MatA = NULL;

    // Allocate the device input vector B
    float *d_MatB = NULL;

    // Allocate the device output vector C
    float *d_MatC = NULL;

    initilization(&h_MatA, &h_MatB, &h_MatC, &d_MatA, &d_MatB, &d_MatC, size, nx, ny, err);

    //execute2D2D(d_MatA, d_MatB, d_MatC, nx, ny, err);

    execute1D1D(d_MatA, d_MatB, d_MatC, nx, ny, err);

    checkMatAdd(h_MatA, h_MatB, h_MatC, d_MatC, nx, ny, size, err);

    //checkHadamardAdd(h_MatA, h_MatB, h_MatC, d_MatC, nx, ny, size, err);

    cleanUp(h_MatA, h_MatB, h_MatC, d_MatA, d_MatB, d_MatC, err);
    return 0;
}

