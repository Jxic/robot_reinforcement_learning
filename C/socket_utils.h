#ifndef SOCKET_UTILS_H
#define SOCKET_UTILS_H

#include "matrix_op.h"
#include "rl_utils.h"

int init_connection(int port);
matrix_t* sim_send(matrix_t* t, int* flag, int state_dim, int act_dim);
void close_connection();


int init_demo_connection(int port);
experience_buffer* build_demo_buffer(int size, int transition_dim);

#endif
