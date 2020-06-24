#ifndef H_COMMON_HEADER_LS
#define H_COMMON_HEADER_LS

float* execute_addition(float* a, float* b, int N, size_t threads, size_t blocks, int deviceId);

float* execute_subtraction(float* a, float* b, int N, size_t threads, size_t blocks, int deviceId);

float* execute_scalar_multiplication(float* a, float k, int N, size_t threads, size_t blocks, int deviceId);

__global__
void addition(float* a, float* b, float* c, int N);

void check_addition(float* a, float *b, float *c, int N);

__global__
void subtraction(float* a, float* b, float* c, int N);

void check_subtraction(float *a, float* b, float* c, int N);

__global__
void scalarmultiplication(float *a, float k, float* c, int N);

void chekc_scalarmultiplication(float *a, float k, float *c, int N);

#endif
