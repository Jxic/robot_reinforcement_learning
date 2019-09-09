#ifndef UTILS_CPP_H
#define UTILS_CPP_H

#include "CL/opencl.h"
#include <string>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
  
void check_status(cl_int status, const char* msg...);
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
    // ~Named_buffer() {
    //   // printf("Reclaiming %s with size %ld\n", name.c_str(), size);
    //   clReleaseMemObject(buffer);
    // }
    string name;
    size_t size;
    cl_mem buffer;
};

class Named_kernel {
  public:
    Named_kernel(const char* name, cl_kernel k) : name(name), k(k) {
      printf("Creating kernel %s\n", name);
    }
    Named_kernel() : name(string("Invalid_kernel")) {}
    string name;
    cl_kernel k;
};

class Named_command_queue {
  public:
    Named_command_queue(const char* name, cl_context c, cl_device_id did, cl_command_queue_properties p) : name(name) {
      printf("Creating named queue for layer %s\n", name);
      cl_int status;
      q = clCreateCommandQueue(c, did, p, &status);
      check_status(status, "Failed creating command queue with name %s", name);
    }
    Named_command_queue() : name(string("Invalid command queue")) {}
    string name;
    cl_command_queue q;
};

Named_buffer find_buffer_by_name(vector<Named_buffer> mem_objs, const char* name);
Named_kernel find_kernel_by_name(vector<Named_kernel> mem_objs, const char* name);
Named_command_queue find_queue_by_name(vector<Named_command_queue> mem_obj, const char* name);


#endif
