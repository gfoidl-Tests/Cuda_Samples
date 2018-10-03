#pragma once
//-----------------------------------------------------------------------------
#include "dll.h"
//-----------------------------------------------------------------------------
BEGIN_EXTERN_C

GPU_API const bool gpu_available();
GPU_API const int gpu_vector_add(double* a, double* b, double* c, const int n);
GPU_API const char* gpu_get_error_string(const int errorCode);

END_EXTERN_C
