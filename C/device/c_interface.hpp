#ifndef C_INTERFACE_HPP
#define C_INTERFACE_HPP

#include "../model.h"
#include "../matrix_op.h"


extern "C" int initialize_training_env(model* m, int batch_size);
extern "C" int free_all_memory_objs();
extern "C" int fpga_forward(model* m, matrix_t* x, matrix_t* y);
extern "C" float fpga_mse_loss_forward(model* m, matrix_t* x, matrix_t* y);
extern "C" matrix_t* retrieve_grad_of_input(model* m, int batch_size);
extern "C" float fpga_dqn_grad(model* m, matrix_t* actions, matrix_t* reward, float gamma);
extern "C" int fpga_transfer_data_to_aux(model* m);
extern "C" int fpga_backward(model* m, matrix_t* grad,int);
extern "C" matrix_t* fpga_adam(model* m, float learning_rate);
extern "C" void initialize_values_on_device(model* m);
extern "C" void print_buffer(const char* buffer_name, int last_n_digit, int type);
extern "C" int fpga_prepare_backward(model* m, int batch_size);
#endif
