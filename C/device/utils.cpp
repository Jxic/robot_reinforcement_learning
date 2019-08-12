#include "utils.hpp"
#include <stdio.h>


void check_status(cl_int status, const char* msg) {
  if (status != CL_SUCCESS) {
    printf("Error: %s\n", msg);
    exit(1);
  }
} 
