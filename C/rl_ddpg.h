#ifndef RL_DDPG_H
#define RL_DDPG_H

#include "model.h"

#define STATE_DIM 3
#define INFO_DIM 2
#define FLAG_DIM 2
#define ACTION_DIM 1
#define UNIT_MOVEMENT 2
#define GAMMA 0.99
#define C_LR 0.001
#define A_LR 0.0001
#define EPOCH 1000
#define POLYAK 0.999
#define MAX_EPOCH_LEN 1000
#define BATCH_SIZE 64 // same as 64 timesteps
#define PRE_TRAIN_STEPS 10000
#define MEMORY_SIZE 100000
#define NOISE_SCALE 0.1
#define RANDOM_INIT_ANGLE 1
#define RANDOM_INIT_DEST 1
#define NUM_OF_LAYERS 3
#define ACTION_BOUND 2
#define ENV_LIMIT 200

#define DDPG_ACTOR_FILE "DDPG_ACTOR_PENDULUM.model"
#define DDPG_ACTOR_T_FILE "DDPG_ACTOR_T_PENDULUM.model"
#define DDPG_CRITIC_FILE "DDPG_CRITIC_PENDULUM.model"
#define DDPG_CRITIC_T_FILE "DDPG_CRITI_T_PENDULUM.model"

void run_ddpg();

#endif
