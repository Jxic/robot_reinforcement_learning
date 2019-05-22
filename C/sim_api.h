#ifndef SIM_API_H
#define SIM_API_H

#include "matrix_op.h"

#define ENV_REACHING 100
#define ENV_PICK_N_PLACE 200

int initEnv(int act_dim, int task_flag);
matrix_t* resetState(int randAngle, int destPos, int state_dim, int act_dim);
matrix_t* step(matrix_t* action, int state_dim, int act_dim);
matrix_t* random_action(int state_dim, int act_dim);
void closeEnv(int state_dim, int act_dim);
void renderSteps(matrix_t** actions, int numOfActions);
matrix_t* inverse_km(matrix_t* ee_pos);



#endif
