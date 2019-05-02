#ifndef RL_H
#define RL_H

#include "model.h"

typedef enum _rl_type {
  test, ddpg
} rl_type;

void run_rl(rl_type t);

#endif
