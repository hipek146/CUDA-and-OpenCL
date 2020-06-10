#include <stdio.h>
#include <chrono>
using namespace std::chrono;

#define N 2048 * 2048 // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively
 * and use profiler to check your progress
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 25us.
 *
 * Some bugs have been placed in this codebase for your edification.
 */

__global__ void initWith(float num, float *a, int Size)
{

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < Size; i += stride)
  {
    a[i] = num;
  }
}

__global__ void saxpy(float * a, float * b, float * c)
{
	 int tid = blockIdx.x * blockDim.x + threadIdx.x;

	    if ( tid < N )
	        c[tid] = 2 * a[tid] + b[tid];
}

int main()
{
	 int deviceId;
	  int numberOfSMs;

	  cudaGetDevice(&deviceId);
	  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    float *a, *b, *c;

    int size = N * sizeof (int); // The total number of bytes per vector

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

     //Initialize memory
//    for( int i = 0; i < N; ++i )
//    {
//        a[i] = 2;
//        b[i] = 1;
//        c[i] = 0;
//    }
    cudaStream_t stream1, stream2, stream3;
      cudaStreamCreate(&stream1);
      cudaStreamCreate(&stream2);
      cudaStreamCreate(&stream3);


    cudaMemPrefetchAsync(a, size, deviceId);
     cudaMemPrefetchAsync(b, size, deviceId);
     cudaMemPrefetchAsync(c, size, deviceId);

     int threads_per_block = 256;
      int number_of_blocks = 32 * numberOfSMs;
      int number_of_blocks_kernel = (N + threads_per_block) / threads_per_block;

     initWith<<<number_of_blocks, threads_per_block, 0, stream1>>>(2, a, N);
     initWith<<<number_of_blocks, threads_per_block, 0, stream2>>>(1, b, N);
     //initWith<<<number_of_blocks, threads_per_block, 0, stream3>>>(0, c, N);


    auto start_fn = high_resolution_clock::now();
    saxpy <<< number_of_blocks_kernel, threads_per_block>>> ( a, b, c);
    auto stop_fn = high_resolution_clock::now();
    auto duration_fn = duration_cast<microseconds>(stop_fn - start_fn);
    cudaDeviceSynchronize();
    cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
    printf("Time: %d us \n", duration_fn);


    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %f, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %f, ", i, c[i]);
    printf ("\n");
    cudaStreamDestroy(stream1);
      cudaStreamDestroy(stream2);
      cudaStreamDestroy(stream3);
    cudaFree( a ); cudaFree( b ); cudaFree( c );
}

