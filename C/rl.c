#include "rl.h"
#include <stdio.h>
#include <stdlib.h>
#include "model.h"

static model* init_rl_model_0();

model* init_rl_model(int version) {
  printf("Using network version %d\n", version);
  switch (version) {
    case 0: {
      return init_rl_model_0();
    }
  
    default:
      printf("[INIT_MODEL] unrecognized model version");
      exit(1);
  }
}

static model* init_rl_model_0() {
  model* new_model = init_model(10);
  add_linear_layer(new_model, 10, relu);
  add_linear_layer(new_model, 50, relu);
  add_linear_layer(new_model, 100, sigmoid);
  add_linear_layer(new_model, 10, placeholder);
  compile_model(new_model, mse_loss);
  print_network(new_model);
  return new_model;
}
