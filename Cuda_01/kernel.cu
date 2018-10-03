#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
//-----------------------------------------------------------------------------
using std::cout;
using std::cerr;
using std::endl;
//-----------------------------------------------------------------------------
__global__
void add(double* x, double* y, const int n)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
//-----------------------------------------------------------------------------
__global__
void root(double* x, const int n)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride)
        x[i] = sqrt(x[i]);
}
//-----------------------------------------------------------------------------
__global__
void shift(int* x, const int n)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride)
        if (i % 2 != 0)
            x[i] = x[i] << 2;
}
//-----------------------------------------------------------------------------
bool checkGPUAvailability()
{
    int deviceCount = 0;
    cudaError_t errorId = cudaGetDeviceCount(&deviceCount);

    if (errorId != cudaSuccess)
    {
        cerr << "cudaGetDeviceCount returned " << static_cast<int>(errorId) << " -> " << cudaGetErrorString(errorId) << endl;
        return false;
    }

    cout << "detected " << deviceCount << " capable device(s) that support CUDA" << endl;
    return true;
}
//-----------------------------------------------------------------------------
void run_add()
{
    const int N = 1 << 20;
    double* x;
    double* y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(double));
    cudaMallocManaged(&y, N * sizeof(double));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; ++i)
    {
        x[i] = 1.0;
        y[i] = 2.0;
    }

    // Run kernel on 1M elements on the GPU
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(x, y, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    double maxError = 0.0f;
    for (int i = 0; i < N; ++i)
        maxError = fmax(maxError, abs(y[i] - 3.0f));
    cout << "Max error: " << maxError << endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
}
//-----------------------------------------------------------------------------
void run_root()
{
    const int N = 1 << 20;
    double* x;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(double));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; ++i)
        x[i] = i;

    // Run kernel on 1M elements on the GPU
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    root<<<numBlocks, blockSize>>>(x, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    double maxError = 0.0f;
    for (int i = 0; i < N; ++i)
        maxError = fmax(maxError, abs(sqrt(i) - x[i]));
    cout << "Max error: " << maxError << endl;

    // Free memory
    cudaFree(x);
}
//-----------------------------------------------------------------------------
void run_shift()
{
    const int N = 1 << 20;
    int* x;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(int));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; ++i)
        x[i] = i;

    // Run kernel on 1M elements on the GPU
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    shift<<<numBlocks, blockSize>>>(x, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    int maxError = 0;
    for (int i = 0; i < N; ++i)
        maxError = fmax(maxError, abs((i << 2) - x[i]));
    cout << "Max error: " << maxError << endl;

    for (int i = 0; i < 10; ++i)
        cout << "\t" << i << "\t" << x[i] << endl;

    // Free memory
    cudaFree(x);
}
//-----------------------------------------------------------------------------
int main(void)
{
    if (!checkGPUAvailability()) return 1;

    cout << "add" << endl;
    cout << "===" << endl;
    run_add();
    
    cout << endl;

    cout << "root" << endl;
    cout << "====" << endl;
    run_root();

    cout << endl;

    cout << "shift" << endl;
    cout << "=====" << endl;
    run_shift();
}
