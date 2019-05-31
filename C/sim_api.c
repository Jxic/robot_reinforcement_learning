#include "sim_api.h"
#include "socket_utils.h"
#include "stdlib.h"
#include <assert.h>
#include "rl_ddpg.h"
#include <stdio.h>
#include "macros.h"
#ifdef MPI
#include "mpi.h"
#endif
#ifndef C_AS_LIB

int initEnv(int act_dim, int task_flag) {
  #ifdef MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  init_connection(6666+rank);
  #else
  init_connection(6666);
  #endif
  return 1;
}

void renderSteps(matrix_t** actions, int numOfActions) {
  printf("C++ sim required\n");
  exit(1);
}
matrix_t* inverse_km(matrix_t* ee_pos) {
  printf("C++ sim required\n");
  exit(1);
}

matrix_t* random_action(int state_dim, int act_dim) {
  matrix_t* init_mat = new_matrix(1, act_dim);
  int* flag = (int*)calloc(2, sizeof(int));
  flag[0] = 0;
  flag[1] = 0;
  matrix_t* ret = sim_send(init_mat, flag, state_dim, act_dim);
  matrix_t* action = slice_col_wise(ret, 0, act_dim);
  free_matrix(init_mat);
  free_matrix(ret);
  return action;
}

matrix_t* resetState(int randAngle, int destPos, int state_dim, int act_dim) {
  int* flag = (int*)calloc(2, sizeof(int));
  flag[0] = 1;
  flag[1] = 0;
  matrix_t* reset_mat = new_matrix(1, act_dim);
  matrix_t* ret = sim_send(reset_mat, flag, state_dim, act_dim);
  free_matrix(reset_mat);
  return ret;
}

matrix_t* step(matrix_t* action, int state_dim, int act_dim) {
  assert(action->cols == act_dim);
  int* flag = (int*)calloc(2, sizeof(int));
  flag[0] = 0;
  flag[1] = 1;
  matrix_t* ret = sim_send(action, flag, state_dim, act_dim);
  return ret;
}

void closeEnv(int state_dim, int act_dim) {
  matrix_t* end_mat = new_matrix(1, act_dim);
  int* flag = (int*)calloc(2, sizeof(int));
  flag[0] = 1;
  flag[1] = 1;
  matrix_t* r = sim_send(end_mat, flag, state_dim, act_dim);
  free_matrix(r);
  free_matrix(end_mat);
  close_connection();
}

#endif

