#include <iostream>
#include "kernels.h"
//-----------------------------------------------------------------------------
using std::cout;
using std::cerr;
using std::endl;
//-----------------------------------------------------------------------------
template<typename T>
T cpu_sum(T* arr, const int n)
{
    T sum = 0;

    for (int i = 0; i < n; ++i)
        sum += arr[i];

    return sum;
}
//-----------------------------------------------------------------------------
template<typename T>
void run()
{
    const int N = 10000;

    T* arr = new T[N];

    for (int i = 0; i < N; ++i)
        arr[i] = i + 1;

    T gpuSum = vec_sum(arr, N);
    T cpuSum = cpu_sum(arr, N);

    cout << "gpu sum: " << gpuSum << endl;
    cout << "cpu sum: " << cpuSum << endl;

    delete[] arr;
}
//-----------------------------------------------------------------------------
void run_atomic(int (*method)(int*, int))
{
    const int N = 10000;

    int* arr = new int[N];

    for (int i = 0; i < N; ++i)
        arr[i] = i + 1;

    int gpuSum = method(arr, N);
    int cpuSum = cpu_sum(arr, N);

    cout << "gpu sum: " << gpuSum << endl;
    cout << "cpu sum: " << cpuSum << endl;

    delete[] arr;
}
//-----------------------------------------------------------------------------
int main()
{
    run<int>();

    cout << endl;

    run_atomic(vec_sum_warp_atomic);
    run_atomic(vec_sum_block_atomic);

    cout << endl;

    run<double>();
}
