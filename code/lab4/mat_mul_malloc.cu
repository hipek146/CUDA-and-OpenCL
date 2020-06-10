#include <stdio.h>
// CUDA sample - elements
// This is not a complete app! Need to add the main section
// The memory is allocated and handled in a 'classical' way
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

void matrixMulCPU( Matrix a, Matrix b, Matrix c, unsigned int N)
{
  int val = 0;

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a.elements[row * N + k] * b.elements[k * N + col];
      c.elements[row * N + col] = val;
    }
}


int main()
{
  Matrix A{BLOCK_SIZE, BLOCK_SIZE, NULL},
  B{BLOCK_SIZE, BLOCK_SIZE, NULL}, C{BLOCK_SIZE, BLOCK_SIZE, NULL},
  C_GPU{BLOCK_SIZE, BLOCK_SIZE, NULL};

  size_t size = A.width * A.height * sizeof(float);
  A.elements = (float*)malloc(size);

  size = B.width * B.height * sizeof(float);
  B.elements = (float*)malloc(size);

  size = C.width * C.height * sizeof(float);
  C.elements = (float*)malloc(size);

  size = C.width * C.height * sizeof(float);
  C_GPU.elements = (float*)malloc(size);


  // Initialize memory
  for( int row = 0; row < BLOCK_SIZE; ++row )
    for( int col = 0; col < BLOCK_SIZE; ++col )
    {
      A.elements[row*BLOCK_SIZE + col] = row;
      B.elements[row*BLOCK_SIZE + col] = col+2;
      C.elements[row*BLOCK_SIZE + col] = 0;
      C_GPU.elements[row*BLOCK_SIZE + col] = 0;
    }

  MatMul(A,B,C_GPU);

  cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

  // Call the CPU version to check our work
  matrixMulCPU( A, B, C, BLOCK_SIZE);

  // Compare the two answers to make sure they are equal
  bool error = false;
  for( int row = 0; row < BLOCK_SIZE && !error; ++row )
    for( int col = 0; col < BLOCK_SIZE && !error; ++col )
      if (C.elements[row * BLOCK_SIZE + col] != C_GPU.elements[row * BLOCK_SIZE + col])
      {
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
  if (!error)
    printf("Success!\n");

  // Free all our allocated memory
  free(A.elements);
  free(B.elements);
  free(C.elements);
  free(C_GPU.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

