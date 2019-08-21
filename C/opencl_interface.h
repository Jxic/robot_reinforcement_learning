#ifndef OPENCL_INTERFACE_H
#define OPENCL_INTERFACE_H
#include "model.h"
#include "matrix_op.h"

int c_init_opencl(int, const char* const*);
int free_all_memory_objs();
float fpga_forward(model*, matrix_t*,matrix_t*);
int fpga_backward(model*, matrix_t*);
int fpga_adam(model*, float);
void initialize_values_on_device(model* );
int initialize_training_env(model*, int);
int fpga_prepare_backward(model* m, int batch_size);


#endif
