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
#include "rl_ddpg_pixel.h"

static void test_run_mse();
static void test_run_cce();
static void test_run_conv();

void run_rl(rl_type t) {
  switch (t)
  {
    case test:
      printf("Running test algorithm ... \n");
      test_run_mse();
      test_run_cce();
      test_run_conv();
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

    case ddpg_pixel:
      printf("Running ddpg on pixel input ... \n");
      run_ddpg_pixel();
    
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

static model* init_model_1(int size) {
  model* new_model = init_model(3);
  add_linear_layer(new_model, size, relu);
  add_linear_layer(new_model, size, relu);
  add_linear_layer(new_model, 4, placeholder);
  compile_model(new_model, cce_loss, adam);
  print_network(new_model);
  return new_model;
}

static model* init_model_2() {
  model* new_model = init_model(28*28*1);
  add_conv_layer(new_model, 28, 28, 1, 6, 5, 1, 2, relu);
  add_max_pool_layer(new_model, 28, 28, 6, 2, 2);
  add_linear_layer(new_model, 120, relu);
  add_linear_layer(new_model, 84, relu);
  add_linear_layer(new_model, 10, placeholder);
  compile_model(new_model, cce_loss, adam);
  print_network(new_model);
  return new_model;
}


void test_run_mse() {
  
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
  int batch_size = 32;
  int epoch = 100;
  double learning_rate = 0.001;
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
  double loss = eval(m, x, y, min_max);
  printf("test run finished with error rate of %f (mse).\n", loss);
}

void test_run_cce() {
  matrix_t* t;
  #ifndef C_AS_LIB
  t = load_data("ROI_dataset.dat");
  #else
  t = load_data("../src/robot_reinforcement_learning/C/FM_dataset.dat");
  #endif
  // matrix_t* min_max = normalize(t);
  shuffle_row_wise(t, 0);  
  matrix_t* x = slice_col_wise(t, 0, 3);
  normalize(x);
  matrix_t* y = slice_col_wise(t, 3, 7);
  
  int batch_size = 32;
  int epoch = 100;
  double learning_rate = 0.001;
  int shuffle = 1;

  model* m;
  m = init_model_1(100);
  fit(m, x, y, batch_size, epoch, learning_rate, shuffle, 1);
  predict(m, x);

  matrix_t* predicted = matrix_row_argmax(x);
  matrix_t* truth = matrix_row_argmax(y);
  int corrected = 0;
  for (int i = 0; i < y->rows; ++i) {
    if (predicted->data[i] == truth->data[i]) {
      corrected++;
    }
  }
  printf("correct %d, accuracy %f\n", corrected, corrected/(double)y->rows);
}

void test_run_conv() {
  matrix_t* t;
  printf("preparing data and model\n");
  #ifndef C_AS_LIB
  t = load_data("train.dat");
  #else
  t = load_data("../src/robot_reinforcement_learning/C/FM_dataset.dat");
  #endif
  // matrix_t* min_max = normalize(t);
  printf("loaded %d lines\n", t->rows);
  // shuffle_row_wise(t, 0);  
  matrix_t* x = slice_col_wise(t, 1, t->cols);
  if (contains_nan(x)) {
    printf("loaded data contains nan\n");
    exit(1);
  }
  matrix_t* temp = normalize(x);
  free_matrix(temp);
  if (contains_nan(x)) {
    printf("normalized data contains nan\n");
    exit(1);
  }
  matrix_t* y = slice_col_wise(t, 0, 1);
  temp = one_hot_encoding(y, 10);
  free_matrix(y);
  y = temp;
  
  int batch_size = 4200;
  int epoch = 20;
  double learning_rate = 0.001;
  int shuffle = 1;

  model* m;
  m = init_model_2();
  print_matrix(t,0);
  print_matrix(y,0);
  print_network(m);

  fit(m, x, y, batch_size, epoch, learning_rate, shuffle, 1);
  predict(m, x);


  matrix_t* predicted = matrix_row_argmax(x);
  matrix_t* truth = matrix_row_argmax(y);
  int corrected = 0;
  for (int i = 0; i < y->rows; ++i) {
    if (predicted->data[i] == truth->data[i]) {
      corrected++;
    }
  }
  printf("correct %d, accuracy %f\n", corrected, corrected/(double)y->rows);

  free_matrix(t);
  free_matrix(x);
  free_matrix(y);
  free_model(m);
  // matrix_t* test_data = load_data("test.dat");
  // normalize(test_data);
  // predict(m, test_data);
  // matrix_t* result = matrix_row_argmax(test_data);
  // FILE* fp =fopen("submission.csv", "w+");
  // for (int i = 0; i < result->rows; ++i) {
  //   fprintf(fp, "%d,%d\n", i+1, (int)result->data[i]);
  // }
  
}
