#include <iostream>
#include <cmath>
#include "gpu_core.h"
//-----------------------------------------------------------------------------
using std::cout;
using std::cerr;
using std::endl;
//-----------------------------------------------------------------------------
int main()
{
    const int N = 1 << 27;      // 1 << 20 = 1M elements

    cout << "array size: " << N << endl;
    
    if (N >= 1 << 28)
    {
        cerr << "array size is >= " << (1 << 28) << " and is too big" << endl;
        exit(-1);
    }

    double* a = new double[N];
    double* b = new double[N];
    double* c = new double[N];

    for (int i = 0; i < N; ++i)
    {
        a[i] = 1.0;
        b[i] = 2.0;
    }

    if (gpu_available())
    {
        cout << "gpu available, data initialized" << endl;
    }
    else
    {
        cerr << "no gpu available" << endl;
        exit(1);
    }

    const int state = gpu_vector_add(a, b, c, N);
    if (state != 0)
    {
        cerr << gpu_get_error_string(state) << endl;
        exit(2);
    }

    double maxError = 0.0;
    for (int i = 0; i < N; ++i)
        maxError = fmax(maxError, abs(c[i] - 3.0));
    cout << endl << "max error: " << maxError << endl;

    delete[] a;
    delete[] b;
    delete[] c;
}
