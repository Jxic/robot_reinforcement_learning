#include "rl.h"
#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "utils.h"

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
  model* new_model = init_model(3);
  add_linear_layer(new_model, 100, relu);
  add_linear_layer(new_model, 100, relu);
  //add_linear_layer(new_model, 100, relu);
  add_linear_layer(new_model, 3, placeholder);
  compile_model(new_model, mse_loss);
  print_network(new_model);
  return new_model;
}

void test_run() {
  model* m = init_rl_model(0);
  matrix_t* t = load_data("FM_dataset.dat");
  matrix_t* min_max = normalize(t);
  print_matrix(min_max, 1);
  shuffle_row_wise(t);
  matrix_t* x = slice_col_wise(t, 0, 3);
  matrix_t* y = slice_col_wise(t, 3, 6);

  int batch_size = 32;
  int epoch = 100;
  double learning_rate = 0.01;
  int shuffle = 0;
  fit(m, x, y, batch_size, epoch, learning_rate, shuffle);
  double loss = eval(m, x, y, min_max);
  printf("test run finished with error rate of %f.\n", loss);
}
