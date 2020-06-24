#ifndef H_VECTOR_HEADER_LS
#define H_VECTOR_HEADER_LS

float* execute_vectordotproduct(float* a, float* b, int N, size_t threads, size_t blocks, int deviceId);

__global__
void createvectortosum(float *a, float *b, float* c, int N);

template<int blocksize>
__global__
void reduce(float *g_data, int N) {
  __shared__ float sdata[2*blocksize];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
  sdata[tid] = g_data[i] + g_data[i + blockDim.x];

  for(unsigned j = blockDim.x; j >0; j >>= 1) {
    __syncthreads();
    if (tid < j) {
      sdata[tid] += sdata[tid + j];
    }
  }

  if (tid == 0) g_data[blockIdx.x] = sdata[tid];
}

void check_vectordotproduct(float *a, float *b, float resutl, int N);

#endif
