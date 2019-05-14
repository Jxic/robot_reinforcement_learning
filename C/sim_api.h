#ifndef SIM_API_H
#define SIM_API_H

#include "matrix_op.h"


int initEnv(int act_dim);
matrix_t* resetState(int randAngle, int destPos, int state_dim, int act_dim);
matrix_t* step(matrix_t* action, int state_dim, int act_dim);
matrix_t* random_action(int state_dim, int act_dim);
void closeEnv(int state_dim, int act_dim);


#endif
