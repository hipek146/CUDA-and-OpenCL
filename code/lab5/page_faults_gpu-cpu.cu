/// managed mamory analysis - cuda lab cpu->gpu only mamory access
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
using namespace std::chrono;

__global__
void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = 1;
  }
}

void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 1;
  }
}

int main(int argc, char** argv)
{
  char* pEnd;
  const int N = 2<<strtol(argv[1], &pEnd, 10); //2<<24;
  size_t size = N * sizeof(int);
  int *a;
  cudaMallocManaged(&a, size);

  FILE *f;
  f = fopen(argv[2], "a");
  if (strtol(argv[1], &pEnd, 10) == 10) {
    fprintf(f, "NumElement\t\tDevice\t\tHost\n");
  }
  fprintf(f, "%d\t\t", N);
  auto start = high_resolution_clock::now();
  deviceKernel<<<256, 256>>>(a, N);
  cudaDeviceSynchronize();
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  fprintf(f, "%d\t\t", duration.count());
  printf("Device: %d us \n", duration.count());
  start = high_resolution_clock::now();
  hostFunction(a, N);
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  fprintf(f, "%d\n", duration.count());
  fclose(f);
  printf("Host: %d us \n", duration.count());
  cudaFree(a);
}
