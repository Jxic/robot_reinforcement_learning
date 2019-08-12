// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

 // ACL kernel for adding two input vectors
__kernel void vector_add(__global const float *x, 
                         __global const float *y, 
                         __global float *restrict z)
{
    // get index of the work item
    int index = get_global_id(0);

    // add the vector elements
    z[index] = x[index] + y[index];
}

__kernel void gemm(__global const float* restrict x, const int x_row_, const int x_col_,
                    __global const float* restrict y, const int y_row_, const int y_col_,
                    __global float* restrict r)
{
    

    __local int x_row; x_row = 100;
    __local int x_col; x_col = 100;
    __local int y_row; y_row = 100;
    __local int y_col; y_col = 100;
    __local int r_row; r_row = 100;
    __local int r_col; r_col = 100;
    
    __local int x_size; x_size = x_row*x_col;
    __local int y_size; y_size = y_row*y_col;
    __local int r_size; r_size = x_row*y_col;


    __local float cache_x[10000];
    __local float cache_y[10000];
    __local float cache_r[10000];

    for (size_t i = 0; i < x_row*x_col; ++i) cache_x[i] = x[i];
    for (size_t i = 0; i < y_row*y_col; ++i) cache_y[i] = y[i];

    #pragma unroll 10
    for (size_t i = 0; i < r_row; ++i) {
        for (size_t k = 0; k < x_col; ++k) {
            // #pragma unroll
            for (size_t j = 0; j < r_col; ++j) {
                cache_r[i*r_col+j] += cache_x[i*x_col+k] * cache_y[k*y_col+j];
            }
        }
    }

    for (size_t i = 0; i < r_row*r_col; ++i) r[i] = cache_r[i]; 
}

