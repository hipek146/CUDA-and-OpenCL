// example of using CUDA streams

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
using namespace std::chrono;

__global__
void initWith(float num, float *a, int N)
{

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    a[i] = num;
  }
}

void initWithNoStream(float num, float *a, int N)
{

  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main(int argc, char** argv)
{
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  char* pEnd;
  const int N = 2<<strtol(argv[1], &pEnd, 10);
  //const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;
  float *d;
  float *e;
  float *f;
  float *g;



  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);
  cudaMallocManaged(&d, size);
  cudaMallocManaged(&e, size);
  cudaMallocManaged(&f, size);
  cudaMallocManaged(&g, size);

  cudaMemPrefetchAsync(a, size, deviceId);
  cudaMemPrefetchAsync(b, size, deviceId);
  cudaMemPrefetchAsync(c, size, deviceId);
  cudaMemPrefetchAsync(d, size, deviceId);
  cudaMemPrefetchAsync(e, size, deviceId);
  cudaMemPrefetchAsync(f, size, deviceId);
  cudaMemPrefetchAsync(g, size, deviceId);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = 32 * numberOfSMs;

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  /*
   * Create 3 streams to run initialize the 3 data vectors in parallel.
   */

  //auto start = high_resolution_clock::now();
  cudaStream_t stream1, stream2, stream3, stream4, stream5, stream6, stream7;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);
  cudaStreamCreate(&stream4);
  cudaStreamCreate(&stream5);
  cudaStreamCreate(&stream6);
  cudaStreamCreate(&stream7);

  /*
   * Give each `initWith` launch its own non-standard stream.
   */

  int which = strtol(argv[2], &pEnd, 10);
  if (which == 0) {
	  initWithNoStream(1.25, a, N);
	  initWithNoStream(1.25, b, N);
	  initWithNoStream(0, c, N);
	  initWithNoStream(0, d, N);
	  initWithNoStream(1.25, e, N);
	  initWithNoStream(0, f, N);
	  initWithNoStream(1.25, g, N);
  }
  if (which == 1) {
	  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream1>>>(3.1, a, N);
	  initWithNoStream(4.25, b, N);
	  initWithNoStream(0, c, N);
  }
  if (which == 2) {
	  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream1>>>(3.1, a, N);
	  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(4.25, b, N);
	  initWithNoStream(0, c, N);
  }
  if (which == 3) {
	  initWithNoStream(3.1, a, N);
	  initWithNoStream(4.25, b, N);
	  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream3>>>(0, c, N);
  }
  if (which == 4) {
	  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream1>>>(1.25, a, N);
	  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(1.25, b, N);
	  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream3>>>(0, c, N);
	  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream4>>>(0, d, N);
	  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream5>>>(1.25, e, N);
	  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream6>>>(0, f, N);
	  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream7>>>(1.25, g, N);
  }
 
  auto start = high_resolution_clock::now();
  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);
  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(d, c, e, N);
  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(f, d, g, N);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  std::cout<< "Time in seconds: " << duration.count()/1E6 << std::endl;

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  cudaMemPrefetchAsync(f, size, cudaCpuDeviceId);

  checkElementsAre(5, f, N);

  /*
   * Destroy streams when they are no longer needed.
   */

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream3);
  cudaStreamDestroy(stream4);
  cudaStreamDestroy(stream5);
  cudaStreamDestroy(stream6);
  cudaStreamDestroy(stream7);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFree(d);
  cudaFree(e);
  cudaFree(f);
  cudaFree(g);
}

