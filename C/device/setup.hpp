#ifndef SETUP_HPP
#define SETUP_HPP

#include <string>
#include <iostream>
#include "CL/opencl.h"
#include <vector>
#include "utils.hpp"


using namespace std;

class Config {
  public:
    Config() {cout << "Creating new configuration ... " << endl;}

    cl_platform_id platform; // 
    string platform_name; //
    string device_name; //
    cl_device_id device_id;//
    cl_context context; //
    string bin_file; //
    cl_program program; //
    vector<cl_command_queue> command_queues;//
    vector<string> kernel_names;
    vector<Named_kernel> kernels;

    vector<Named_buffer> mem_objs;

    void show();
};

Config init_opencl(vector<string> kernel_names);
extern "C" int c_init_opencl(int, const char* const*);
int free_opencl();
// int test_kernels();

#endif
