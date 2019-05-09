#ifndef SIM_API_H
#define SIM_API_H

#include "matrix_op.h"


int initEnv();
matrix_t* resetState(int randAngle, int destPos);
matrix_t* step(matrix_t* action);
double random_action();
void closeEnv();


#endif
