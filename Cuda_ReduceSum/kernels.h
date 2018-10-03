#pragma once
//-----------------------------------------------------------------------------
#include "cuda_runtime.h"
//-----------------------------------------------------------------------------
int vec_sum(int* vec, const int n);
int vec_sum_warp_atomic(int* vec, const int n);
int vec_sum_block_atomic(int* vec, const int n);
double vec_sum(double* vec, const int n);
