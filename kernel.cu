#include<iostream>
#include<fstream>
#include<stdio.h>
#include<stdlib.h>
#include <iomanip>
#include <Windows.h>
#include"cuda_runtime.h"
#include"cuda.h"
#include"device_launch_parameters.h"
#include<immintrin.h>  
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
using namespace std;
const int N = 500;
const int BLOCK_SIZE = 1024;
float elm[N][N] = { 0 };

void C_GE(float** a, int n)
{
	float t1, t2;
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
		for (int j = k + 1; j < n; j++)
			a[k][j] = a[k][j] / t1;
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (int j = k + 1; j < n; j++)
				a[i][j] -= t2 * a[k][j];
			a[i][k] = 0;
		}
	}
}
__global__ void div_kernel(float* data, int k, int N)
{
	int tid = threadIdx.x;
	float element = data[k * N + k];
	while (k + tid + 1 < N)
	{
		data[k * (N + 1) + tid + 1] /= element;
		tid += blockDim.x;
	}
	return;
}
__global__ void elim_kernel(float* data, int k, int N)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	if (!tx) 
		data[k * N + k] = 1.0;
	int row = k + 1 + blockIdx.x;
	float t;
	while (row < N)
	{
		int tid = threadIdx.x;
		t = data[(row * N) + k];
		while (k + 1 + tid < N)
		{
			int col = k + 1 + tid;
			data[(row * N) + col] = data[(row * N) + col] - t * data[k * N + col];
			tid = tid + blockDim.x;
		}
		__syncthreads();
		//块内同步
		if (threadIdx.x == 0) 
			data[row * N + k] = 0;
		row += gridDim.x;
	}
	return;
}
float** generate(int n)
{
	ifstream inp("input.txt");
	inp >> n;
	float** m = new float* [n];
	for (int i = 0; i < n; i++)
	{
		m[i] = new float[n];
		for (int j = 0; j < n; j++) 
			inp >> m[i][j];
	}
	inp.close();
	return m;
}
float* generate_1d(int n)
{
	ifstream inp("input.txt"); inp >> n;
	float* m = new float[n * n];
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			inp >> m[i * n + j];
	inp.close();
	return m;
}
void CUDA_GE(float* in)
{
	cudaError_t ret;
	float* gpudata;
	float* result = new float[N * N];
	int size = N * N * sizeof(float);
	if (cudaMalloc(&gpudata, size) != cudaSuccess)  
		printf("cudaMalloc gpudata failed!\n");
	if (cudaMemcpy(gpudata, in, size, cudaMemcpyHostToDevice) != cudaSuccess) 
		printf("cudaMemcpyHostToDevice failed!\n");
	dim3 dimBlock(BLOCK_SIZE, 1), dimGrid(1, 1);
	cudaEvent_t start, stop; 
	float elapsedTime = 0.0;
	cudaEventCreate(&start), cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaError_t exec;
	for (int k = 0; k < N; k++)
	{
		div_kernel << <dimGrid, dimBlock >> > (gpudata, k, N);
		cudaDeviceSynchronize();
		exec = cudaGetLastError();
		if (exec != cudaSuccess) 
			printf("division_kernel failed, %s\n", cudaGetErrorString(exec));
		elim_kernel << <dimGrid, dimBlock >> > (gpudata, k, N);
		cudaDeviceSynchronize();
		exec = cudaGetLastError();
		if (exec != cudaSuccess) 
			printf("eliminate_kernel failed, %s\n", cudaGetErrorString(exec));
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("CUDA_GE:%f ms\n", elapsedTime);
	cudaError_t cudaStatus2 = cudaGetLastError();
	if (cudaGetLastError() != cudaSuccess) 
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus2));
	if (cudaMemcpy(result, gpudata, size, cudaMemcpyDeviceToHost) != cudaSuccess) 
		printf("cudaMemcpyDeviceToHost failed!\n");
	cudaFree(gpudata);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
void CUDA_GE_Opt(float* in)
{
	cudaError_t ret;
	float* gpudata;
	float* result = new float[N * N];
	int size = N * N * sizeof(float);
	if (cudaMallocManaged(&gpudata, size) != cudaSuccess)  
		printf("cudaMalloc gpudata failed!\n");
	if (cudaMemcpy(gpudata, in, size, cudaMemcpyHostToDevice) != cudaSuccess) 
		printf("cudaMemcpyHostToDevice failed!\n");
	int deviceId;
	int numberOfSMs;
	int my_cudaDevAttrConcurrentManagedAccess;
	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	int number_of_blocks = numberOfSMs;  
	int threads_per_block = 1024;  
	cudaEvent_t start, stop;  
	float elapsedTime = 0.0;
	cudaEventCreate(&start), cudaEventCreate(&stop);
	cudaEventRecord(start, 0); 
	cudaError_t exec;
	for (int k = 0; k < N; k++)
	{
		div_kernel << <number_of_blocks, threads_per_block >> > (gpudata, k, N);
		cudaDeviceSynchronize();
		exec = cudaGetLastError();
		if (exec != cudaSuccess) 
			printf("division_kernel failed, %s\n", cudaGetErrorString(exec));
		elim_kernel << <number_of_blocks, threads_per_block >> > (gpudata, k, N);
		cudaDeviceSynchronize();
		exec = cudaGetLastError();
		if (exec != cudaSuccess) 
			printf("eliminate_kernel failed, %s\n", cudaGetErrorString(exec));
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);//停止计时
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("CUDA_GE_Opt:%f ms\n", elapsedTime);
	cudaError_t cudaStatus2 = cudaGetLastError();
	if (cudaGetLastError() != cudaSuccess) 
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus2));
	if (cudaMemcpy(result, gpudata, size, cudaMemcpyDeviceToHost) != cudaSuccess) 
		printf("cudaMemcpyDeviceToHost failed!\n");
	cudaFree(gpudata);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
int main()
{
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	float** m1 = generate(N);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	C_GE(m1, N);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "C_GE: " << (tail - head) * 1000.0 / freq<< "ms" << endl;
	float* t_1d = generate_1d(N);
	CUDA_GE(t_1d);
	t_1d = generate_1d(N);
	CUDA_GE_Opt(t_1d);
}
