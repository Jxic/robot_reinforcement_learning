#include "rl.h"
#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "utils.h"
#include "macros.h"
#include "model_utils.h"
#include "rl_ddpg.h"
#include "rl_ddpg_her.h"
#include "rl_ddpg_her_sim.h"
#include "rl_ddpg_her_demo.h"
#include "rl_ddpg_her_demo_sim.h"
#include "multi_agents/rl_ddpg_her_mpi.h"

static void test_run();

void run_rl(rl_type t) {
  switch (t)
  {
    case test:
      printf("Running test algorithm ... \n");
      test_run();
      break;
    
    // deep deterministic policy gradient
    case ddpg:
      printf("Running ddpg ... \n");
      run_ddpg();
      break;

    case her:
      printf("Running ddpg with hindsight experience replay ... \n");
      run_ddpg_her();
      break;

    case her_demo:
      printf("Running ddpg with HER and demo ...\n");
      run_ddpg_her_w_demo();
      break;

    case her_sim:
      printf("Running ddpg with her on C++ simulation ... \n");
      run_rl_ddpg_her_sim();

    case her_demo_sim:
      printf("Running ddpg with her & demo on C++ simulation ... \n");
      run_ddpg_her_w_demo_sim();

    case her_mpi:
      printf("Running ddpg with her in MPI mode ... \n");
      run_ddpg_her_mpi();
    
    default:
      printf("[RUN_MODEL] unrecognized model %d", t);
  }
}

static model* init_model_0(int size) {
  model* new_model = init_model(3);
  new_model->version = 0;
  add_linear_layer(new_model, size, relu);
  add_linear_layer(new_model, size, relu);
  add_linear_layer(new_model, 3, placeholder);
  compile_model(new_model, mse_loss, adam);
  print_network(new_model);
  return new_model;
}


void test_run() {
  
  matrix_t* t;
  #ifndef C_AS_LIB
  t = load_data("FM_dataset.dat");
  #else
  t = load_data("../src/robot_reinforcement_learning/C/FM_dataset.dat");
  #endif
  matrix_t* min_max = normalize(t);
  shuffle_row_wise(t, 0);
  matrix_t* x = slice_col_wise(t, 0, 3);
  matrix_t* y = slice_col_wise(t, 3, 6);
  int batch_size = 16;
  int epoch = 100;
  float learning_rate = 0.001;
  int shuffle = 1;

  model* m;
  //for (int i = 100; i < 1000; i+=100) {
    m = init_model_0(100);
    fit(m, x, y, batch_size, epoch, learning_rate, shuffle, 1);
    //free_model(m);
  //}
  
  // save_model(m, "test_model.model");
  // model* m_ = load_model("test_model.model");
  // print_network(m_);
  float loss = eval(m, x, y, min_max);
  printf("test run finished with error rate of %f (mse).\n", loss);
}
