/// impact of prefetching the data - CUDA lab

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
using namespace std::chrono;

void initWith(float num, float *a, int N)
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
  printf("Device ID: %d\tNumber of SMs: %d\n", deviceId, numberOfSMs);

  char* pEnd;
  const int N = 2<<strtol(argv[1], &pEnd, 10); //2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  FILE *f;
  f = fopen(argv[2], "a");
  if (strtol(argv[1], &pEnd, 10) == 10) {
    fprintf(f, "NumElement\t\tInit\t\tDevice\n");
  }
  fprintf(f, "%d\t\t", N);

  auto start = high_resolution_clock::now();
  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  printf("Init: %d us \n", duration.count());
  fprintf(f, "%d\t\t", duration.count());

  /*
   * Add asynchronous prefetching after the data is initialized,
   * and before launching the kernel, to avoid host to GPU page
   * faulting.
   */

  cudaMemPrefetchAsync(a, size, deviceId);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = 32 * numberOfSMs;

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  start = high_resolution_clock::now();
  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  printf("Device: %d us \n", duration.count());
  fprintf(f, "%d\n", duration.count());
  fclose(f);

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

