#pragma once
//-----------------------------------------------------------------------------
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//-----------------------------------------------------------------------------
__global__ void addKernel(double* a, double* b, double* c, const int n);
