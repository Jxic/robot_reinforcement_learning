#include "rl.h"
#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "utils.h"
#include "macros.h"
#include "layers.h"
#include "model_utils.h"
#include "rl_ddpg.h"
#include "rl_ddpg_her.h"
#include "rl_ddpg_her_sim.h"
#include "rl_ddpg_her_demo.h"
#include "rl_ddpg_her_demo_sim.h"
#include "multi_agents/rl_ddpg_her_mpi.h"
#include "rl_ddpg_pixel.h"
#include <math.h>


static void test_run_mse();
static void test_run_cce();
static void test_run_conv();
static void test_device();

void run_rl(rl_type t) {
  switch (t)
  {
    case test:
      printf("Running test algorithm ... \n");
      // test_run_mse();
      // test_run_cce();
      // test_run_conv();
      test_device();
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
    #ifdef MPI
    case her_mpi:
      printf("Running ddpg with her in MPI mode ... \n");
      run_ddpg_her_mpi();
    #endif
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
  t = load_data("dat/FM_dataset.dat");
  #else
  t = load_data("../src/robot_reinforcement_learning/C/FM_dataset.dat");
  #endif
  matrix_t* min_max = normalize(t);
  shuffle_row_wise(t, 0);
  matrix_t* x = slice_col_wise(t, 0, 3);
  matrix_t* y = slice_col_wise(t, 3, 6);
  int batch_size = 32;
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

void test_run_cce() {
  matrix_t* t;
  #ifndef C_AS_LIB
  t = load_data("dat/ROI_dataset.dat");
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
  float learning_rate = 0.001;
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
  printf("correct %d, accuracy %f\n", corrected, corrected/(float)y->rows);
}

void test_run_conv() {
  matrix_t* t;
  printf("preparing data and model\n");
  #ifndef C_AS_LIB
  t = load_data("dat/train.dat");
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
  int epoch = 30;
  float learning_rate = 0.001;
  int shuffle = 1;

  model* m;
  m = init_model_2();
  // print_matrix(t,0);
  // print_matrix(y,0);
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
  printf("correct %d, accuracy %f\n", corrected, corrected/(float)y->rows);

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

#ifdef OPENCL
#include "opencl_interface.h"
static void test_device() {
  matrix_t* t;
  #ifndef C_AS_LIB
  t = load_data("dat/FM_dataset.dat");
  #else
  t = load_data("../src/robot_reinforcement_learning/C/FM_dataset.dat");
  #endif
  matrix_t* min_max = normalize(t);
  shuffle_row_wise(t, 0);
  matrix_t* x = slice_col_wise(t, 0, 3);
  matrix_t* y = slice_col_wise(t, 3, 6);
  int batch_size = 32;
  int epoch = 100;
  float learning_rate = 0.0001;
  int shuffle = 1;

  model* m;
  m = init_model_0(100);
  const char * names[] = {
    // "vector_add",
    // "gemm",
    "linear_forward_prop",
    "relu_forward_prop",
    "mse",
    "relu_backward_prop",
    "transpose_params_n_cache",
    "linear_backward_prop",
    "generate_update_adam",
    "examine_int_array",
    "examine_float_array",
    "transpose_params_n_cache",
    "matmul_engine",
    #ifdef USING_CHANNEL
    "channel_start",
    "channel_end",
    "channel_manager",
    "prepare_input_grads",
    "b_channel_end",
    "b_channel_manager",
    #endif
  };
  int num_of_kernels = 11;
  #ifdef USING_CHANNEL
  num_of_kernels = 16;
  #endif
  c_init_opencl(num_of_kernels, names);
  initialize_training_env(m, batch_size);
  initialize_values_on_device(m);
  printf("done preparing device\n");

  matrix_t* ex = slice_row_wise(x, 0, 32);
  matrix_t* ey = slice_row_wise(y, 0, 32);
  matrix_t* d_ex = matrix_clone(ex);
  matrix_t* d_ey = matrix_clone(ey);

  
  fpga_forward(m, d_ex, d_ey);
  float loss_device = fpga_mse_loss_forward(m, d_ex, d_ey);

  printf("done device side forward\n");
  predict(m, ex);
  float loss_host = loss_forward(&m->loss_layer, ex, ey);
  printf("done host side forward\n");
  printf("loss host: %f loss device: %f\n", loss_host, loss_device);

  printf("===================================\n");

  // fpga_backward(m, 0.01);
  fpga_prepare_backward(m, batch_size);
  fpga_backward(m, new_matrix(1,1), 0);
  retrieve_grad_of_input(m, batch_size);


  matrix_t* grad_host = loss_backward(&m->loss_layer );
  model_backward(m, grad_host);
  matrix_t* gh = new_matrix(1, m->param_size);
  for (int i = 0; i < m->param_size; ++i) gh->data[i] = *(m->opt.cache.a.trainable_params_g[i]);
  gh->cols = 30;
  printf("host grads\n");
  // print_matrix(gh, 1);
  grad_host->rows = 1;
  grad_host->cols = 30;
  print_matrix(grad_host, 1);

  printf("====================================\n");

  matrix_t* pd = fpga_adam(m, learning_rate);
  perform_update(m, learning_rate);
  matrix_t* ph = new_matrix(1, m->param_size);
  for (int i = 0; i < m->param_size; ++i) ph->data[i] = *(m->opt.cache.a.trainable_params[i]);
  float sum = 0;
  float ld = 0;
  for (int i = 0; i < m->param_size; ++i) {
    float difference = fabs(pd->data[i]-ph->data[i]);
    if (difference > ld) {
      ld = difference;
    }
    sum += difference;
  }
  printf("largest difference %e\n", ld);
  printf("total difference on updated parameter: %e\n", sum);
  exit(1);
  printf("====================================\n");
  printf("Second round\n");

  matrix_t* ex_2 = slice_row_wise(x, 32, 64);
  matrix_t* ey_2 = slice_row_wise(y, 32, 64);
  matrix_t* d_ex_2 = matrix_clone(ex_2);
  matrix_t* d_ey_2 = matrix_clone(ey_2);

  loss_device = fpga_forward(m, d_ex_2, d_ey_2);
  printf("done device side forward\n");
  predict(m, ex_2);
  loss_host = loss_forward(&m->loss_layer, ex_2, ey_2);
  printf("done host side forward\n");
  printf("loss host: %f loss device: %f\n", loss_host, loss_device);

  printf("===================================\n");

  // fpga_backward(m, 0.01);
  fpga_prepare_backward(m, batch_size);
  fpga_backward(m, new_matrix(1,1), 0);

  matrix_t* grad_host_2 = loss_backward(&m->loss_layer );
  model_backward(m, grad_host_2);
  matrix_t* gh_2 = new_matrix(1, m->param_size);
  for (int i = 0; i < m->param_size; ++i) gh_2->data[i] = *(m->opt.cache.a.trainable_params_g[i]);
  gh_2->cols = 30;
  print_matrix(gh_2, 1);

  printf("====================================\n");

  matrix_t* pd_2 = fpga_adam(m, learning_rate);
  perform_update(m, learning_rate);
  matrix_t* ph_2 = new_matrix(1, m->param_size);
  for (int i = 0; i < m->param_size; ++i) ph_2->data[i] = *(m->opt.cache.a.trainable_params[i]);
  float sum_2 = 0;
  float ld_2 = 0;
  for (int i = 0; i < m->param_size; ++i) {
    float difference = fabs(pd_2->data[i]-ph_2->data[i]);
    if (difference > ld_2) {
      ld_2 = difference;
    }
    sum_2 += difference;
  }
  printf("largest difference %e\n", ld_2);
  printf("total difference on updated parameter: %e\n", sum_2);
  
  printf("====================================\n");

  free_all_memory_objs();
  // // float loss = eval(m, x, y, min_max);
  // // printf("test run finished with error rate of %f (mse).\n", loss);
  // matrix_t* cache_T = transpose(m->hidden_linears[0].data.l.cache);
  // cache_T->rows = 1;
  // cache_T->cols = 50;
  // print_matrix(cache_T,1);
}
#endif