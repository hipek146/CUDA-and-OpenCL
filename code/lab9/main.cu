// example of using CUDA streams

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include <fstream>
using namespace std::chrono;

#define BLOCK_SIZE 256

void initWithNoStream(float num, float *a, int N)
{

  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void reduceVector(float *g_data, int N) {
  __shared__ float sdata[2*BLOCK_SIZE];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
  unsigned int gridStride = 2 * blockDim.x * gridDim.x;
  sdata[tid] = 0;
  while (i < N) {
	  sdata[tid] += g_data[i] + g_data[i + blockDim.x];
	  i += gridStride;
  }
  for(unsigned j = blockDim.x / 2; j > 32; j >>= 1) {
	  __syncthreads();
	  if (tid < j) {
		sdata[tid] += sdata[tid + j];
	  }
	}
  //printf("tid %d i %d blockIdx.x %d sdata[%d] %f\n", tid, i, blockIdx.x, tid, sdata[tid]);

  __syncthreads();
//
  if(tid < 32) {
	  sdata[tid] += sdata[tid + 32];
	  sdata[tid] += sdata[tid + 16];
	  sdata[tid] += sdata[tid + 8];
	  sdata[tid] += sdata[tid + 4];
	  sdata[tid] += sdata[tid + 2];
	  sdata[tid] += sdata[tid + 1];
  }
  if (tid == 0) g_data[blockIdx.x] = sdata[0];
}

__global__
void reduceVector_naive(float *g_data, int N) {
  __shared__ float sdata[2*BLOCK_SIZE];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
  sdata[tid] = g_data[i];
  sdata[blockDim.x+tid] = g_data[i + blockDim.x];


  for(unsigned int j = 1; j <= blockDim.x; j *= 2) {
    __syncthreads();
    if (tid % j == 0) {
      sdata[2*tid] += sdata[2*tid + j];
    }
  }

  __syncthreads();
  if (tid == 0) g_data[blockIdx.x] = sdata[tid];
}

int main(int argc, char** argv)
{
  int in_s = 14;
  char* pEnd;
  if (!in_s && argc < 2) {
    printf("Podaj liczbe która będzie przesunięciem 2 które jest rozmiarem\n");
    return 1;
  } else if(!in_s) {
	  in_s = strtol(argv[1], &pEnd, 10);
  }
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  std::ofstream host_file, device_file, device_naive_file;
  host_file.open ("reduction_host.txt");
  device_file.open ("reduction_device.txt");
  device_naive_file.open ("reduction__device_naive.txt");

  for(int in_s = 14; in_s < 27; in_s++) {
	  const int N = 2<<in_s;
	  host_file << N;
	  device_file << N;
	  device_naive_file << N;
	  size_t size = N * sizeof(float);


	  size_t threads;
	  size_t blocks;

	  threads = BLOCK_SIZE;
	  blocks = N / threads / 2;

	  printf("threads %d blocks %d\n", threads, blocks);

	  cudaError_t addVectorsErr;
	  cudaError_t asyncErr;

	  for(int k = 0; k < 10; k++) {
		  float *a;
		  float *b;
		  float *c;

		  cudaMallocManaged(&b, size);
		  cudaMallocManaged(&a, size);
		  cudaMallocManaged(&c, size);

		  initWithNoStream(1, a, N);
		  initWithNoStream(1, b, N);
		  initWithNoStream(1, c, N);

		  cudaMemPrefetchAsync(a, size, deviceId);
		  cudaMemPrefetchAsync(b, size, deviceId);
		  asyncErr = cudaDeviceSynchronize();
			if(asyncErr != cudaSuccess) printf("Device sync Error: %s\n", cudaGetErrorString(asyncErr));

		  auto start = high_resolution_clock::now();
		  reduceVector<<<32 * numberOfSMs, threads>>>(a, N);

		  cudaMemPrefetchAsync(a, size, cudaCpuDeviceId);
		  asyncErr = cudaDeviceSynchronize();
		  if(asyncErr != cudaSuccess) printf("Device sync Error: %s\n", cudaGetErrorString(asyncErr));

		  for(int i = 1; i < 32 * numberOfSMs; i++) {
				a[0] += a[i];
			  }
		  auto stop = high_resolution_clock::now();
		  auto duration = duration_cast<microseconds>(stop - start);
		  std::cout<< "Reduction time in us: " << duration.count()<< std::endl;
		  device_file << "\t" << duration.count();

		  addVectorsErr = cudaGetLastError();
		  if(addVectorsErr != cudaSuccess) printf("Reduction Error: %s\n", cudaGetErrorString(addVectorsErr));


		  start = high_resolution_clock::now();
		  reduceVector_naive<<<blocks, threads>>>(b, N);
		  cudaMemPrefetchAsync(b, size, cudaCpuDeviceId);

		  asyncErr = cudaDeviceSynchronize();
		  if(asyncErr != cudaSuccess) printf("Device sync Error: %s\n", cudaGetErrorString(asyncErr));

		  for(int i = 1; i < blocks; i++) {
			  b[0] += b[i];
			}
		  stop = high_resolution_clock::now();
		  duration = duration_cast<microseconds>(stop - start);
		  std::cout<< "Reduction (naive) time in us: " << duration.count()<< std::endl;
		  device_naive_file << "\t" << duration.count();

		  addVectorsErr = cudaGetLastError();
		  if(addVectorsErr != cudaSuccess) printf("Reduction Error: %s\n", cudaGetErrorString(addVectorsErr));

		  asyncErr = cudaDeviceSynchronize();
		  if(asyncErr != cudaSuccess) printf("Device sync Error: %s\n", cudaGetErrorString(asyncErr));


		  double result = 0;
		  start = high_resolution_clock::now();
		  for(long i = 0; i < N; i++) {
			result += c[i];
		  }
		  stop = high_resolution_clock::now();
		  duration = duration_cast<microseconds>(stop - start);
		  std::cout<< "Cpu add rest time in us: " << duration.count() << std::endl;
		  host_file << "\t" << duration.count();

		  printf("%f %f %f %d\n", a[0], b[0], result, N);

		  cudaFree(a);
		  cudaFree(b);
		  cudaFree(c);
	  }
  }
}

