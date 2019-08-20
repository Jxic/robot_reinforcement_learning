#include "tests.hpp"
#include "setup.hpp"
#include <vector>
#include <string>
#include "CL/opencl.h"
#include "utils.hpp"
#include "math.h"
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

Config conf;

void gemm_test();

void kernel_tests(vector<string> kns) {
  conf = init_opencl(kns);
  gemm_test();
}

float rand_float(float low, float high) {
  assert(high >= low);
  return float(rand()) / float(RAND_MAX) * (high - low) + low;
}

void gemm_test() {
  printf("Running GEMM ...\n");
  cl_int status;

  int a_row = 100;
  int a_col = 100;
  int b_row = 100;
  int b_col = 100;
  int r_row = a_row;
  int r_col = b_col;

  size_t mat_a_size = a_row * a_col;
  size_t mat_b_size = b_row * b_col;
  size_t mat_c_size = r_row * r_col;

  float mat_a_host[mat_a_size];
  float mat_b_host[mat_b_size];
  float mat_c_host[mat_c_size];

  cl_context context = conf.context;
  cl_command_queue  default_q = conf.command_queues[0];
  cl_kernel vector_add = conf.kernels[0].k;
  cl_kernel gemm = conf.kernels[1].k;

  cl_event write_event[2];
  cl_event kernel_event[1];
  cl_event read_event[1];

  cl_mem mat_a = clCreateBuffer(context, CL_MEM_READ_WRITE, mat_a_size*sizeof(float), NULL, &status);
  check_status(status, "Failed creating buffer for matrix A");
  cl_mem mat_b = clCreateBuffer(context, CL_MEM_READ_WRITE, mat_b_size*sizeof(float), NULL, &status);
  check_status(status, "Failed creating buffer for matrix B");
  cl_mem mat_c = clCreateBuffer(context, CL_MEM_READ_WRITE, mat_c_size*sizeof(float), NULL, &status);
  check_status(status, "Failed creating buffer for return value");


  for (size_t i = 0 ; i < mat_a_size; i++) mat_a_host[i] = rand_float(-10, 10);
  for (size_t i = 0 ; i < mat_b_size; i++) mat_b_host[i] = rand_float(-10, 10);

  status = clEnqueueWriteBuffer(default_q, mat_a, CL_FALSE, 0, mat_a_size*sizeof(float), mat_a_host, 0, NULL, write_event);
  check_status(status, "Failed transferring data to device(a)");
  status = clEnqueueWriteBuffer(default_q, mat_b, CL_FALSE, 0, mat_b_size*sizeof(float), mat_b_host, 0, NULL, write_event+1);
  check_status(status, "Failed transferring data to device(b)");

  // set kernel args  
  size_t arg_idx = 0;
  status = clSetKernelArg(gemm, arg_idx++, sizeof(cl_mem), &mat_a);
  check_status(status, "Failed set arg 0");
  status = clSetKernelArg(gemm, arg_idx++, sizeof(int), &a_row);
  check_status(status, "Failed set arg 1");
  status = clSetKernelArg(gemm, arg_idx++, sizeof(int), &a_col);
  check_status(status, "Failed set arg 2");

  status = clSetKernelArg(gemm, arg_idx++, sizeof(cl_mem), &mat_b);
  check_status(status, "Failed set arg 3");
  status = clSetKernelArg(gemm, arg_idx++, sizeof(int), &b_row);
  check_status(status, "Failed set arg 4");
  status = clSetKernelArg(gemm, arg_idx++, sizeof(int), &b_col);
  check_status(status, "Failed set arg 5");

  status = clSetKernelArg(gemm, arg_idx++, sizeof(cl_mem), &mat_c);
  check_status(status, "Failed set arg 6");

  // run kernel
  const size_t gws = 1;
  status = clEnqueueNDRangeKernel(default_q, gemm, 1, NULL, &gws, NULL, 2, write_event, kernel_event);

  status = clEnqueueReadBuffer(default_q, mat_c, CL_FALSE, 0, mat_c_size*sizeof(float), mat_c_host, 1, kernel_event, read_event);
  check_status(status, "Failed reading data from device(a)");

  clWaitForEvents(1, read_event);

  // verification
  printf("Verification ... \n");
  // float* mat_c_prime_host = new float[mat_c_size];
  float mat_c_prime_host[mat_c_size];
  for (int i = 0; i < r_row*r_col; ++i) mat_c_prime_host[i] = 0;
  for (int i = 0; i < r_row; ++i) {
        for (int k = 0; k < a_col; ++k) {
            for (int j = 0; j < r_col; ++j) {
                mat_c_prime_host[i*r_col+j] += mat_a_host[i*a_col+k] * mat_b_host[k*b_col+j];
            }
        }
  }

  for (size_t i = 0; i < mat_c_size; ++i) {
    if (fabsf(mat_c_prime_host[i] - mat_c_host[i]) > 1.0e-5f) {
      printf("Failed i: %ld | %f %f\n", i, mat_c_prime_host[i], mat_c_host[i]);
      exit(1);
    }
  }
  printf("Passed\n");
}
