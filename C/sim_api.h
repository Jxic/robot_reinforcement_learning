#ifndef SIM_API_H
#define SIM_API_H

#include "matrix_op.h"

int initEnv();
matrix_t* resetState(int randAngle, int destPos);
matrix_t* step(matrix_t* action);

#ifndef __cplusplus
int initEnv() {
  return 1;
}

matrix_t* resetState(int randAngle, int destPos) {
  return new_matrix(1, 10);
}

matrix_t* step(matrix_t* action) {
  return new_matrix(1, 10);
}
#endif

#endif
