#include "setup.hpp"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.hpp"
#include <algorithm>

#define PLATFORM_NAME "Intel(R) FPGA SDK for OpenCL(TM)"
#define BIN_NAME "rl.aocx"


int c_init_opencl() {
  printf("called from c\n");
  vector<string> kns;
  kns.push_back("vector_add");
  kns.push_back("gemm");
  Config c = init_opencl(kns);
  c.show();
  return 1;
}


void oclContextCallback(const char *errinfo, const void *, size_t, void *) {
  printf("Context callback: %s\n", errinfo);
}


Config init_opencl(vector<string> kernel_names) {
  Config new_conf;
  cl_int status;

  new_conf.platform_name = string(PLATFORM_NAME);
  string p_name(PLATFORM_NAME);
  transform(p_name.begin(), p_name.end(), p_name.begin(), ::tolower);

  cout << "starting" << endl;
  // Get platform
  cl_uint num_platforms;
  status = clGetPlatformIDs(0, NULL, &num_platforms);
  assert(num_platforms == 1);
  check_status(status ,"Failed finding number of platforms");

  cl_platform_id* ids = new cl_platform_id[num_platforms];
  status = clGetPlatformIDs(num_platforms, ids, NULL);
  check_status(status, "Failed finding all platform ids");
  cl_platform_id pid = *ids;
  delete[] ids;
  new_conf.platform = pid;

  
  size_t p_sz;
  status = clGetPlatformInfo(pid, CL_PLATFORM_NAME, 0, NULL, &p_sz);
  check_status(status, "Failed finding platform name size");
  char* p_name_buffer = new char[p_sz];
  status = clGetPlatformInfo(pid, CL_PLATFORM_NAME, p_sz, p_name_buffer, NULL);
  string searched_name = string(p_name_buffer);
  delete[] p_name_buffer;
  transform(searched_name.begin(), searched_name.end(), searched_name.begin(), ::tolower);
  if (searched_name.find(p_name) == string::npos) {printf("Name not matched\n"); exit(1);}

  // Get device
  cl_uint num_devices;
  status = clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  printf("Number of devices %d\n", num_devices);
  assert(num_devices == 1);
  check_status(status, "Failed finding number of devices");

  cl_device_id* dids = new cl_device_id[num_devices];
  status = clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, num_devices, dids, NULL);
  check_status(status, "Failed finding device ids");
  cl_device_id did = *dids;
  delete[] dids;
  new_conf.device_id = did;
  
  size_t sz;
  status = clGetDeviceInfo(did, CL_DEVICE_NAME, 0, NULL, &sz);
  check_status(status, "Failed finding device name size");

  char* name_buffer = new char[sz];
  status = clGetDeviceInfo(did, CL_DEVICE_NAME, sz, name_buffer, NULL);
  check_status(status, "Failed finding devive name");
  new_conf.device_name = string(name_buffer);
  delete[] name_buffer;

  // create context
  new_conf.context = clCreateContext(NULL, num_devices, &did, &oclContextCallback, NULL, &status);
  check_status(status, "Failed creating context");

  //create program
  size_t bin_size;
  FILE* fp;
  fp = fopen(BIN_NAME, "rb");
  if (!fp) { cout << "Failed opening binary" << endl; exit(1);}
  fseek(fp, 0, SEEK_END);
  bin_size = ftell(fp);
  unsigned char* binary = new unsigned char[bin_size];
  rewind(fp);
  if (fread((void*)binary, bin_size, 1, fp) == 0) {
    delete[] binary;
    fclose(fp);
    cout << "Failed loading binary" << endl;
    exit(1);
  }
  std::string file_name = std::string(BIN_NAME);
  new_conf.bin_file = file_name;
  new_conf.program = clCreateProgramWithBinary(new_conf.context, num_devices, &did, &bin_size,
    (const unsigned char **)&binary, &status, &status);
  check_status(status, "Failed loading binary for device");

  //build program
  status = clBuildProgram(new_conf.program, 0, NULL, "", NULL, NULL);
  check_status(status, "Failed building program");

  //create command queue, default to 1
  cl_command_queue default_queue = clCreateCommandQueue(new_conf.context, did, CL_QUEUE_PROFILING_ENABLE, &status);
  check_status(status, "Failed creating command queue");
  new_conf.command_queues.push_back(default_queue);

  //initialize kernels
  for (size_t i = 0; i < kernel_names.size(); i++) {
    new_conf.kernel_names.push_back(kernel_names[i]);
    new_conf.kernels.push_back(clCreateKernel(new_conf.program, kernel_names[i].c_str(), &status));
    check_status(status, "Failed initializing kernel");
  }

  //done
  cout << "Created new configuration ... " << endl;
  return new_conf;
}

void Config::show() {
  cout << "-----------Configuration-----------" << endl;

  cout << "Platform name: " << this->platform_name << endl;
  cout << "Platform id:   " << this->platform << endl;
  cout << "Device name:   " << this->device_name << endl;
  cout << "Device id:     " << this->device_id << endl;
  cout << "Binary file:   " << this->bin_file << endl;
  cout << "Command Queue: " << this->command_queues.size() << " asynch queues" << endl;
  cout << "Kernels:       " << this->kernel_names.size() << " kernel(s)" << endl;
  for (size_t i = 0; i < this->kernel_names.size(); i++) {
    cout << " | " << kernel_names[i] << endl;
  }
  cout << "-----------------------------------" << endl;
}


