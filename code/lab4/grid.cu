
// CUDA sample
// simple grid-stride

#include <stdio.h>
#include <cstdlib>

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

__global__
void doubleElementsStride(int *a, int N)
{

  /*
   * Use a grid-stride loop so each thread does work
   * on more than one element in the array.
   */

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] *= 2;
  }
}

__global__
void doubleElementsMismatch(int *a, int N)
{

  /*
   * Use a grid-stride loop so each thread does work
   * on more than one element in the array.
   */

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] *= 2;
  }
}

bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (a[i] != i*2) return false;
  }
  return true;
}

int main(int argc, char** argv)
{
  int N = 10000;
  int *a;
  int blocks = 32;
  int threads = 256;
  int flag = 0;
  if(argc > 3) {
	  N = atoi(argv[1]);
	  threads = atoi(argv[2]);
	  blocks = atoi(argv[3]);
	  if(argc > 4) {
		  flag = atoi(argv[4]);
	  }
  }

  size_t size = N * sizeof(int);
  cudaMallocManaged(&a, size);

  size_t threads_per_block = threads;
  size_t number_of_blocks = blocks;
  bool areDoubled;

  if(flag == 0 || flag == 1) {
	  init(a, N);
	  doubleElementsStride<<<number_of_blocks, threads_per_block>>>(a, N);
	  cudaDeviceSynchronize();

	  areDoubled = checkElementsAreDoubled(a, N);
	  printf("All elements were doubled(stride-grid)? %s\n", areDoubled ? "TRUE" : "FALSE");
  }
  if(flag == 0 || flag == 2) {
	  init(a, N);

	  size_t threads_per_block_mismatch = 256;
	  size_t number_of_blocks_mismatch = (N + threads_per_block_mismatch) / threads_per_block_mismatch;

	  doubleElementsMismatch<<<number_of_blocks_mismatch, threads_per_block_mismatch>>>(a, N);
	  cudaDeviceSynchronize();

	  areDoubled = checkElementsAreDoubled(a, N);
	  printf("All elements were doubled(mismatch)? %s\n", areDoubled ? "TRUE" : "FALSE");
  }
  cudaFree(a);
}

