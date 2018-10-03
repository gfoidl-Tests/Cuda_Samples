#include "gpu_core.h"
#include <cuda_runtime.h>
#include "kernels.cuh"

#if defined(DEBUG) || defined(_DEBUG)
    #include <stdio.h>
    #include <assert.h>
#endif
//-----------------------------------------------------------------------------
// Forward declarations
int getNumSMs();
inline cudaError_t checkCuda(cudaError_t result);
//-----------------------------------------------------------------------------
const bool gpu_available()
{
    int deviceCount;
    cudaError_t errorId = cudaGetDeviceCount(&deviceCount);

    return errorId == cudaSuccess
        && deviceCount > 0;
}
//-----------------------------------------------------------------------------
const int gpu_vector_add(double* a, double* b, double* c, const int n)
{
    double* dA;
    double* dB;
    double* dC;

    const int size = sizeof(double) * n;

    try
    {
        checkCuda(cudaMalloc(&dA, size));
        checkCuda(cudaMalloc(&dB, size));
        checkCuda(cudaMalloc(&dC, size));

        checkCuda(cudaMemcpy(dA, a, size, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(dB, b, size, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(dC, c, size, cudaMemcpyHostToDevice));

        const int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;

#if defined(DEBUG) || defined(_DEBUG)
        printf("blockSize: %d\nnumBlocks: %d\n", blockSize, numBlocks);
#endif

        addKernel<<< numBlocks, blockSize >>> (dA, dB, dC, n);

        checkCuda(cudaMemcpy(c, dC, size, cudaMemcpyDeviceToHost));

        checkCuda(cudaDeviceSynchronize());

        checkCuda(cudaFree(dA));
        checkCuda(cudaFree(dB));
        checkCuda(cudaFree(dC));
    }
    catch (const int e)
    {
        return e;
    }

    return 0;
}
//-----------------------------------------------------------------------------
const char* gpu_get_error_string(const int errorCode)
{
    return cudaGetErrorString(static_cast<cudaError>(errorCode));
}
//-----------------------------------------------------------------------------
int getNumSMs()
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    return numSMs;
}
//-----------------------------------------------------------------------------
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));

        assert(result == cudaSuccess);
    }
#else
    if (result != cudaSuccess)
    {
        throw static_cast<int>(result);
    }
#endif

    return result;
}
