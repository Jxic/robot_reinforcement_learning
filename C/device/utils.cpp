#include "utils.hpp"
#include <stdio.h>
#include <stdarg.h>

void check_status(cl_int status, const char* msg...) {
    if(status != CL_SUCCESS) {
    // Print line and file
    printf("ERROR[%d]: ", status);
    // printf("Location: %s:%d\n", __FILE__, __LINE__);

    va_list vl;
    va_start(vl, msg);
    vprintf(msg, vl);
    printf("\n");
    va_end(vl);

    exit(1);
  }
} 

Named_buffer find_buffer_by_name(vector<Named_buffer> mem_objs, const char* name) {
  string s_name = string(name);
  for (size_t i = 0; i < mem_objs.size(); ++i) {
    if (s_name.compare(mem_objs[i].name) == 0) {
      return mem_objs[i];
    }
  }
  cout << "Cannot find buffer with given name " << s_name << ", exiting ..." << endl;
  exit(1);
  return Named_buffer();
}

Named_kernel find_kernel_by_name(vector<Named_kernel> mem_objs, const char* name) {
  string s_name = string(name);
  for (size_t i = 0; i < mem_objs.size(); ++i) {
    if (s_name.compare(mem_objs[i].name) == 0) {
      return mem_objs[i];
    }
  }
  cout << "Cannot find kernel with given name " << s_name << ", exiting ..." << endl;
  exit(1);
  return Named_kernel();
}

Named_command_queue find_queue_by_name(vector<Named_command_queue> mem_objs, const char* name) {
  string s_name = string(name);
  for (size_t i = 0; i < mem_objs.size(); ++i) {
    if (s_name.compare(mem_objs[i].name) == 0) {
      return mem_objs[i];
    }
  }
  cout << "Cannot find kernel with given name " << s_name << ", exiting ..." << endl;
  exit(1);
  return Named_command_queue();
}
