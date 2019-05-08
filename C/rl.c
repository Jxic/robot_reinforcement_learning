#include "rl.h"
#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "utils.h"
#include "macros.h"
#include "model_utils.h"
#include "rl_ddpg.h"


static void test_run();

void run_rl(rl_type t) {
  switch (t)
  {
    case test:
      test_run();
      break;
    
    // deep deterministic policy gradient
    case ddpg:
      run_ddpg();
      break;
    
    default:
      printf("[RUN_MODEL] unrecognized model %d", t);
  }
}

static model* init_model_0() {
  model* new_model = init_model(3);
  new_model->version = 0;
  add_linear_layer(new_model, 100, relu);
  add_linear_layer(new_model, 100, relu);
  add_linear_layer(new_model, 3, placeholder);
  compile_model(new_model, mse_loss, adam);
  print_network(new_model);
  return new_model;
}


void test_run() {
  model* m = init_model_0();
  matrix_t* t;
  #ifndef C_AS_LIB
  t = load_data("FM_dataset.dat");
  #else
  t = load_data("./src/robot_reinforcement_learning/C/FM_dataset.dat");
  #endif

  matrix_t* min_max = normalize(t);

  shuffle_row_wise(t, 0);
  matrix_t* x = slice_col_wise(t, 0, 3);
  matrix_t* y = slice_col_wise(t, 3, 6);

  int batch_size = 16;
  int epoch = 100;
  double learning_rate = 0.001;
  int shuffle = 1;
  fit(m, x, y, batch_size, epoch, learning_rate, shuffle);
  save_model(m, "test_model.model");
  model* m_ = load_model("test_model.model");
  init_caches(m_, x->rows);
  print_network(m_);
  double loss = eval(m_, x, y, min_max);
  printf("test run finished with error rate of %f (mse).\n", loss);
}
