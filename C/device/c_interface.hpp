#ifndef C_INTERFACE_HPP
#define C_INTERFACE_HPP

#include "../model.h"
#include "../matrix_op.h"


extern "C" int initialize_training_env(model* m, int batch_size);
extern "C" int free_all_memory_objs();
extern "C" float fpga_forward(model* m, matrix_t* x, matrix_t* y);
extern "C" int fpga_backward(model* m, matrix_t* grad);
extern "C" int fpga_adam(model* m, float learning_rate);
extern "C" void initialize_values_on_device(model* m);
extern "C" void print_buffer(const char* buffer_name, int last_n_digit, int type);
extern "C" int fpga_prepare_backward(model* m, int batch_size);
#endif
