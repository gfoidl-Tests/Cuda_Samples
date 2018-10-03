using System;
using System.Runtime.InteropServices;
using static System.Math;

namespace GpuTestNet
{
    class Program
    {
        static void Main(string[] args)
        {
            const int N = 1 << 27;      // 1 << 20 = 1M elements

            Console.WriteLine($"array size: {N}");
#if !DEBUG
            if (N >= 1 << 28)
            {
                Console.Error.WriteLine($"array size is >= {1 << 28} and is too big");
                Environment.Exit(-1);
            }
#endif
            double[] a = new double[N];
            double[] b = new double[N];
            double[] c = new double[N];

            for (int i = 0; i < N; ++i)
            {
                a[i] = 1d;
                b[i] = 2d;
            }

            if (Gpu.IsAvailable())
            {
                Console.WriteLine("gpu available, data initialized");
            }
            else
            {
                Console.Error.WriteLine("no gpu available");
                Environment.Exit(1);
            }

            try
            {
                Gpu.VectorAdd(a, b, c);
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.Error.WriteLine(ex.Message);
                Console.ResetColor();
            }

            double maxError = 0d;
            foreach (double val in c)
                maxError = Max(maxError, Abs(val - 3d));
            Console.WriteLine($"\nmax error: {maxError}");
        }
    }
    //-------------------------------------------------------------------------
    internal static class Gpu
    {
        private const string DllName = "GpuCore";
        //---------------------------------------------------------------------
        [DllImport(DllName, EntryPoint = "gpu_available")]
        public static extern bool IsAvailable();
        //---------------------------------------------------------------------
        [DllImport(DllName, EntryPoint = "gpu_vector_add")]
        private static unsafe extern int VectorAdd(double* a, double* b, double* c, int n);
        //---------------------------------------------------------------------
        [DllImport(DllName, EntryPoint = "gpu_get_error_string")]
        private static extern IntPtr GetErrorString(int errorCode);
        //---------------------------------------------------------------------
        public static unsafe void VectorAdd(double[] a, double[] b, double[] c)
        {
            fixed (double* pA = a)
            fixed (double* pB = b)
            fixed (double* pC = c)
            {
                int state = VectorAdd(pA, pB, pC, a.Length);

                if (state != 0)
                {
                    IntPtr ptr = GetErrorString(state);
                    string msg = $"CUDA Runtime Error: {Marshal.PtrToStringAnsi(ptr)}";
                    throw new Exception(msg);
                }
            }
        }
    }
}
