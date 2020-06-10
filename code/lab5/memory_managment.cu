///
/// CUDA lab, AGH course 2020 summer
/// This code is especially prepared by NVIDIA team for studying
/// memory mamagement techniques. We are going to use it as working
/// template for iterative optimisation.
///

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
using namespace std::chrono;

/*
 * Host function to initialize vector elements. This function
 * simply initializes each element to equal its index in the
 * vector.
 */

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

/*
 * Device kernel stores into `result` the sum of each
 * same-indexed value of `a` and `b`.
 */

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

/*
 * Host function to confirm values in `vector`. This function
 * assumes all values are the same `target` value.
 */

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
    fprintf(f, "NumElement\t\tInit\t\tDevice\t\tHost\n");
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

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  /*
   * nsys should register performance changes when execution configuration
   * is updated.
   */

  threadsPerBlock = 1;
  numberOfBlocks = 1;

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
  fprintf(f, "%d\t\t", duration.count());

  start = high_resolution_clock::now();
  checkElementsAre(7, c, N);
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  printf("Host: %d us \n", duration.count());
  fprintf(f, "%d\n", duration.count());
  fclose(f);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

