#ifndef SIM_API_H
#define SIM_API_H

#include "matrix_op.h"
#include "socket_utils.h"
#include "stdlib.h"
#include <assert.h>
#include "rl_ddpg.h"

int initEnv();
matrix_t* resetState(int randAngle, int destPos);
matrix_t* step(matrix_t* action);

#ifndef C_AS_LIB

int initEnv() {
  init_connection();
  return 1;
}

double random_action() {
  matrix_t* init_mat = new_matrix(1, ACTION_DIM);
  int* flag = (int*)calloc(2, sizeof(int));
  flag[0] = 0;
  flag[1] = 0;
  matrix_t* ret = sim_send(init_mat, flag);
  double r = ret->data[0];
  free_matrix(init_mat);
  free_matrix(ret);
  return r;
}

matrix_t* resetState(int randAngle, int destPos) {
  int* flag = (int*)calloc(2, sizeof(int));
  flag[0] = 1;
  flag[1] = 0;
  matrix_t* reset_mat = new_matrix(1, ACTION_DIM);
  matrix_t* ret = sim_send(reset_mat, flag);
  free_matrix(reset_mat);
  return ret;
}

matrix_t* step(matrix_t* action) {
  assert(action->cols == ACTION_DIM);
  int* flag = (int*)calloc(2, sizeof(int));
  flag[0] = 0;
  flag[1] = 1;
  matrix_t* ret = sim_send(action, flag);
  return ret;
}

void closeEnv() {
  matrix_t* end_mat = new_matrix(1, ACTION_DIM);
  int* flag = (int*)calloc(2, sizeof(int));
  flag[0] = 1;
  flag[1] = 1;
  matrix_t* r = sim_send(end_mat, flag);
  free_matrix(r);
  free_matrix(end_mat);
  close_connection();
}

#endif

#endif
