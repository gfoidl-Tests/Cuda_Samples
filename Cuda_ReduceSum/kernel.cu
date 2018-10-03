#include "kernels.h"
#include <device_launch_parameters.h>
#include <algorithm>
//-----------------------------------------------------------------------------
__inline__
__device__
int warpReduceSum(int value)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        value += __shfl_down(value, offset);

    return value;
}
//-----------------------------------------------------------------------------
__inline__
__device__
double warpReduceSum(double value)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        value += __shfl_down(value, offset);

    return value;
}
//-----------------------------------------------------------------------------
__inline__
__device__
int blockReduceSum(int value)
{
    static __shared__ int shared[32];
    const int lane = threadIdx.x % warpSize;
    const int warpId = threadIdx.x / warpSize;

    value = warpReduceSum(value);

    if (lane == 0)
        shared[warpId] = value;

    __syncthreads();

    value = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (warpId == 0) value = warpReduceSum(value);

    return value;
}
//-----------------------------------------------------------------------------
__inline__
__device__
double blockReduceSum(double value)
{
    static __shared__ double shared[32];
    const int lane = threadIdx.x % warpSize;
    const int warpId = threadIdx.x / warpSize;

    value = warpReduceSum(value);

    if (lane == 0)
        shared[warpId] = value;

    __syncthreads();

    value = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (warpId == 0) value = warpReduceSum(value);

    return value;
}
//-----------------------------------------------------------------------------
__global__ void deviceReduceSumKernel(int* in, int* out, const int n)
{
    int sum = 0;

    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride)
        sum += in[i];

    sum = blockReduceSum(sum);

    if (threadIdx.x == 0)
        out[blockIdx.x] = sum;
}
//-----------------------------------------------------------------------------
__global__ void deviceReduceSumWarpAtomicKernel(int* in, int* out, const int n)
{
    int sum = 0;

    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride)
        sum += in[i];

    sum = warpReduceSum(sum);

    if (threadIdx.x % warpSize == 0)
        atomicAdd(out, sum);
}
//-----------------------------------------------------------------------------
__global__ void deviceReduceSumBlockAtomicKernel(int* in, int* out, const int n)
{
    int sum = 0;

    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride)
        sum += in[i];

    sum = blockReduceSum(sum);

    if (threadIdx.x == 0)
        atomicAdd(out, sum);
}
//-----------------------------------------------------------------------------
__global__ void deviceReduceSumKernel(double* in, double* out, const int n)
{
    double sum = 0;

    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride)
        sum += in[i];

    sum = blockReduceSum(sum);

    if (threadIdx.x == 0)
        out[blockIdx.x] = sum;
}
//-----------------------------------------------------------------------------
template<typename T>
T vec_sum_core(T* vec, const int n)
{
    const int size = sizeof(T) * n;

    T* dVec;
    T* dOut;

    cudaMalloc(&dVec, size);
    cudaMemcpy(dVec, vec, size, cudaMemcpyHostToDevice);

    const int threads = 512;
    const int blocks = std::min((n + threads - 1) / threads, 1024);

    cudaMalloc(&dOut, sizeof(T) * blocks);

    deviceReduceSumKernel << <blocks, threads >> > (dVec, dOut, n);
    deviceReduceSumKernel << <1, 1024 >> > (dOut, dOut, blocks);

    cudaDeviceSynchronize();

    T sum;
    cudaMemcpy(&sum, dOut, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(dVec);
    cudaFree(dOut);

    return sum;
}
//-----------------------------------------------------------------------------
int vec_sum(int* vec, const int n)
{
    return vec_sum_core(vec, n);
}
//-----------------------------------------------------------------------------
int vec_sum_warp_atomic(int* vec, const int n)
{
    const int size = sizeof(int) * n;

    int* dVec;
    int* dOut;

    cudaMalloc(&dVec, size);
    cudaMalloc(&dOut, sizeof(int));
    cudaMemcpy(dVec, vec, size, cudaMemcpyHostToDevice);

    const int threads = 512;
    const int blocks = std::min((n + threads - 1) / threads, 1024);

    cudaMemsetAsync(dOut, 0, sizeof(int));
    deviceReduceSumWarpAtomicKernel<<< blocks, threads >>>(dVec, dOut, n);
    
    cudaDeviceSynchronize();

    int sum;
    cudaMemcpy(&sum, dOut, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dVec);
    cudaFree(dOut);

    return sum;
}
//-----------------------------------------------------------------------------
int vec_sum_block_atomic(int* vec, const int n)
{
    const int size = sizeof(int) * n;

    int* dVec;
    int* dOut;

    cudaMalloc(&dVec, size);
    cudaMalloc(&dOut, sizeof(int));
    cudaMemcpy(dVec, vec, size, cudaMemcpyHostToDevice);

    const int threads = 512;
    const int blocks = std::min((n + threads - 1) / threads, 1024);

    cudaMemsetAsync(dOut, 0, sizeof(int));
    deviceReduceSumBlockAtomicKernel<<< blocks, threads >>>(dVec, dOut, n);

    cudaDeviceSynchronize();

    int sum;
    cudaMemcpy(&sum, dOut, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dVec);
    cudaFree(dOut);

    return sum;
}
//-----------------------------------------------------------------------------
double vec_sum(double* vec, const int n)
{
    return vec_sum_core(vec, n);
}
