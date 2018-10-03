#include "kernels.cuh"
//-----------------------------------------------------------------------------
__global__
void addKernel(double* a, double* b, double* c, const int n)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride)
        c[i] = a[i] + b[i];
}
