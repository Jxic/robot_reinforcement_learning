#ifndef UTILS_CPP_H
#define UTILS_CPP_H

#include "CL/opencl.h"
#include <string>
#include <iostream>
#include <vector>
#include <stdlib.h>

using namespace std;
  
void check_status(cl_int status, const char* msg);
class Named_buffer {
  public:
  //clCreateBuffer(context, CL_MEM_READ_WRITE, m->param_size*sizeof(float), NULL, &status);
    Named_buffer(const char* name, cl_context context, cl_mem_flags flag, size_t size, void* host_ptr) : name(name), size(size) {
      cl_int status;
      string buffer_info =  string(name) + string(" with size: ") + to_string(size);
      cout << "Creating named buffer " << buffer_info << endl;
      buffer = clCreateBuffer(context, flag, size, host_ptr, &status);
      string err_msg = string("Failed creating named buffer: ") + buffer_info;
      check_status(status, err_msg.c_str());
    }
    Named_buffer() : name(string("Invalid_buffer")) {}
    string name;
    size_t size;
    cl_mem buffer;
};

class Named_kernel {
  public:
    Named_kernel(const char* name, cl_kernel k) : name(name), k(k) {}
    Named_kernel() : name(string("Invalid_kernel")) {}
    string name;
    cl_kernel k;
};

Named_buffer find_buffer_by_name(vector<Named_buffer> mem_objs, const char* name);
Named_kernel find_kernel_by_name(vector<Named_kernel> mem_objs, const char* name);



#endif
