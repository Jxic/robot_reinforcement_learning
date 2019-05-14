#ifndef SOCKET_UTILS_H
#define SOCKET_UTILS_H

#include "matrix_op.h"

int init_connection();
matrix_t* sim_send(matrix_t* t, int* flag, int state_dim, int act_dim);
void close_connection();

#endif
