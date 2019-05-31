#ifndef RL_H
#define RL_H

#include "model.h"

typedef enum _rl_type {
  test, ddpg, her, her_sim, her_demo, her_demo_sim, her_mpi
} rl_type;

void run_rl(rl_type t);

#endif
