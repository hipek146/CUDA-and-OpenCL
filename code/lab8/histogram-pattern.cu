#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <fstream>
using namespace std::chrono;

#define nBins 2024


__global__ void
hist_device(const int *input, int *bins, int N)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	__shared__ unsigned int hist[nBins];
	for (int i = threadIdx.x; i < nBins; i += blockDim.x)
	{
		hist[i] = 0;
	}
	__syncthreads();

	for(int i = index; i < N; i += stride)
	{
		atomicAdd(&hist[input[i]], 1);
	}
	__syncthreads();
	for (int i = threadIdx.x; i < nBins; i += blockDim.x)
	{
		bins[i + blockIdx.x * nBins] = hist[i];
	}
}

__global__ void
hist_device_saturation(const int *input, int *bins, int N)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = index; i < nBins; i += stride)
	{
		unsigned int sum = 0;
		for(int k = 0; k < gridDim.x; k++)
		{
			sum += bins[i + k * nBins];
		}
		bins[i] = sum;
	}
}

__global__ void
hist_device_simple(const int *input, int *bins, int N)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = index; i < N; i += stride)
	{
		atomicAdd(&bins[input[i]], 1);
	}
}

void hist_host(const int *input, int *bins, int N)
{
    for (int i = 0; i < N; i++)
    {
        bins[input[i]]++;
    }
}

int main()
{
    srand(time(NULL));

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    int *bins;
    int *input;

    int threads_per_block = 256;
    int number_of_blocks = 32 * numberOfSMs;

    std::ofstream host_file, device_file, device_simple_file;

	host_file.open ("histogram_host.txt");
	device_file.open ("histogram_device.txt");
	device_simple_file.open ("histogram_device_simple.txt");
    for(int it = 3; it <=9; it++)
    {
		const int N = pow(10, it);
		std::cout << "\nN: " << N << std::endl;
		size_t size = N * sizeof (int);
        size_t size_bins = number_of_blocks * nBins * sizeof (int);

        cudaMallocManaged(&input, size);
        cudaMallocManaged(&bins, size_bins);
        host_file << N;
        device_file << N;
        device_simple_file << N;

        for(int k = 0; k < 10; k++)
        {
            std::cout << "It: " << k + 1 << std::endl;
            for (int i = 0; i < nBins; i++)
            {
                bins[i] = 0;
            }
            for (int i = 0; i < N; i++)
            {
                input[i] = rand() % nBins;
            }


            auto start = high_resolution_clock::now();

            hist_host(input, bins, N);

            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            std::cout << "Host time: " << duration.count() << std::endl;
            std::cout << "Last bin count: " << bins[nBins - 1] << std::endl;
            host_file << "\t" << duration.count();

            //reset histogram
            for (int i = 0; i < nBins; i++)
            {
                bins[i] = 0;
            }

            cudaMemPrefetchAsync(input, size, deviceId);
            cudaMemPrefetchAsync(bins, size_bins, deviceId);

            start = high_resolution_clock::now();

            hist_device <<< number_of_blocks, threads_per_block>>> (input, bins, N);
            hist_device_saturation <<< number_of_blocks, threads_per_block>>> (input, bins, N);
            cudaDeviceSynchronize();

            stop = high_resolution_clock::now();
            duration = duration_cast<microseconds>(stop - start);
            std::cout << "Device time: " << duration.count() << std::endl;

            cudaMemPrefetchAsync(bins, size_bins, cudaCpuDeviceId);
            cudaMemPrefetchAsync(input, size, cudaCpuDeviceId);
            std::cout << "Last bin count: " << bins[nBins - 1] << std::endl;
            device_file << "\t" << duration.count();
            cudaDeviceSynchronize();
            //reset histogram
            for (int i = 0; i < nBins; i++)
            {
                bins[i] = 0;
            }

            cudaMemPrefetchAsync(input, size, deviceId);
            cudaMemPrefetchAsync(bins, size_bins, deviceId);

            start = high_resolution_clock::now();

            hist_device_simple <<< number_of_blocks, threads_per_block>>> (input, bins, N);
            cudaDeviceSynchronize();

            stop = high_resolution_clock::now();
            duration = duration_cast<microseconds>(stop - start);
            std::cout << "Device simple time: " << duration.count() << std::endl;

            cudaMemPrefetchAsync(bins, size_bins, cudaCpuDeviceId);
            cudaMemPrefetchAsync(input, size, cudaCpuDeviceId);
            std::cout << "Last bin count: " << bins[nBins - 1] << std::endl;
            device_simple_file << "\t" << duration.count();

            std::cout << std::endl;
        }
        host_file << std::endl;
        device_file << std::endl;
        device_simple_file << std::endl;
        cudaFree(input);
        cudaFree(bins);
    }
	host_file.close();
	device_file.close();
    return 0;
}
