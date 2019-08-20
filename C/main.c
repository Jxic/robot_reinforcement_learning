#include "matrix_op.h"
#include <stdio.h>
#include <stdlib.h>
#include "tests.h"
#include "macros.h"
#include "utils.h"
#include "rl.h"
#include <time.h>
#include <string.h>
#include "normalizer.h"
#include "model_utils.h"
#include "test_agent.h"
#include "socket_utils.h"
#include "sim_api.h"
#include "layers.h"
#ifdef MKL
#include "mkl.h"
#endif


#ifdef OPENCL
#include "opencl_interface.h"
#endif

#ifdef RUN_TEST
static int simple_test();
#endif

int conv_forward_test() {
  matrix_t* a = new_matrix(2, 8);
  for (int i = 0; i < 16; ++i) a->data[i] = i;
  augment_space(a, 1000, 1000);
  print_matrix(a, 1);
  layer la;
  la.type = conv;
  
  la.data.l.W = new_matrix(8, 2);
  la.data.l.b = new_matrix(1, 2);
  la.data.l.cache = new_matrix(1,1);
  initialize(la.data.l.b, zeros);
  for (int i = 0; i < 16; ++i) la.data.l.W->data[i] = i + 1;
  print_matrix(la.data.l.W, 1);
  la.data.l.sizes[0] = 2;
  la.data.l.sizes[1] = 2;
  la.data.l.sizes[2] = 2;

  la.data.l.input_sizes[0] = 2;
  la.data.l.input_sizes[1] = 2;
  la.data.l.input_sizes[2] = 2;

  la.data.l.stride = 1;
  la.data.l.padding = 1;
  conv_forward(&la, a);
  print_matrix(a,1);
  return 0;
}

int recon_test() {
  matrix_t* a = new_matrix(2,8);
  for (int i = 0; i < 16; ++i) a->data[i] = i;
  matrix_t* recon = conv_reconstruct_input(a, 2,2,2,2,2,2,1,1);
  printf("recon\n");
  print_matrix(a, 1);
  print_matrix(recon, 1);
  printf("reverse\n");
  layer la;
  la.type = conv;
  
  la.data.l.W = new_matrix(8, 2);
  la.data.l.b = new_matrix(1, 2);
  la.data.l.cache = new_matrix(1,1);
  initialize(la.data.l.b, zeros);
  for (int i = 0; i < 16; ++i) la.data.l.W->data[i] = i + 1;
  print_matrix(la.data.l.W, 1);
  la.data.l.sizes[0] = 2;
  la.data.l.sizes[1] = 2;
  la.data.l.sizes[2] = 2;

  la.data.l.input_sizes[0] = 2;
  la.data.l.input_sizes[1] = 2;
  la.data.l.input_sizes[2] = 2;

  la.data.l.stride = 1;
  la.data.l.padding = 1;
  update_grad_x(recon, a, &la);
  print_matrix(a, 1);
  
  // matrix_t* unpad_a = unpad(a, 2,2,2,1);
  // print_matrix(unpad_a, 1);

  return 0;
}

int conv_backward_test() {
  matrix_t* a = new_matrix(2, 8);
  for (int i = 0; i < 16; ++i) a->data[i] = i;
  augment_space(a, 1000, 1000);
  print_matrix(a, 1);
  layer la;
  la.type = conv;
  
  la.data.l.W = new_matrix(8, 2);
  la.data.l.b = new_matrix(1, 2);
  matrix_t* grad = new_matrix(2, 18);
  la.data.l.grad_b = new_matrix(1,2);
  la.data.l.grad_W = new_matrix(8,2);
  initialize(la.data.l.grad_b, zeros);
  initialize(la.data.l.grad_W, zeros);

  for (int i = 0; i < 36; ++i) grad->data[i] = 1;
  augment_space(grad, 1000,1000);
  la.data.l.cache = new_matrix(1,1);
  initialize(la.data.l.b, zeros);
  for (int i = 0; i < 16; ++i) la.data.l.W->data[i] = i + 1;
  print_matrix(la.data.l.W, 1);
  la.data.l.sizes[0] = 2;
  la.data.l.sizes[1] = 2;
  la.data.l.sizes[2] = 2;

  la.data.l.input_sizes[0] = 2;
  la.data.l.input_sizes[1] = 2;
  la.data.l.input_sizes[2] = 2;

  la.data.l.stride = 1;
  la.data.l.padding = 1;
  conv_forward(&la, a);
  conv_backward(&la, grad);
  printf("grad_x\n");
  print_matrix(grad, 1);
  printf("grad_b\n");
  print_matrix(la.data.l.grad_b,1);
  print_matrix(la.data.l.grad_W,1);
  printf("a\n");
  print_matrix(a,1);
  return 0;
}

int max_pool_test() {
  matrix_t* a = new_matrix(2, 32);
  for (int i = 0; i < 64; ++i) a->data[i] = i;
  augment_space(a, 1000, 1000);
  print_matrix(a, 1);
  layer la;
  la.type = max_pool;
  
  la.data.max.cache = new_matrix(1,1);
  la.data.max.sizes[0] = 2;
  la.data.max.sizes[1] = 2;
  la.data.max.sizes[2] = 2;

  la.data.max.input_sizes[0] = 4;
  la.data.max.input_sizes[1] = 4;
  la.data.max.input_sizes[2] = 2;

  la.data.max.stride = 2;
  printf("pooling\n");
  max_pool_forward(&la, a);
  print_matrix(a,1);
  print_matrix(la.data.max.cache, 1);
  printf("grad\n");
  matrix_t* grad = new_matrix(2, 8);
  for (int i = 0; i < 16; ++i) grad->data[i] = 1;
  augment_space(grad, 100, 100);
  max_pool_backward(&la, grad);
  print_matrix(grad, 1);
  printf("done\n");
  return 0;
}

int _main() {
  // preparation phase
  srand(SEED);

  #ifdef MKL
  #ifdef MULTI_MKL_THREAD
  mkl_set_num_threads(0);
  #endif
  #endif

  #ifdef RUN_TEST 
  return simple_test();
  // return conv_forward_test();
  // return recon_test();
  // return conv_backward_test();
  // return max_pool_test();
  #else
            
  // matrix_t* a = new_matrix(3,4);
  // matrix_t* b = new_matrix(4,5);
  // for (int i = 0; i < 12; ++i) {
  //   a->data[i] = (float) i;
  // }
  // for (int i = 0; i < 20; ++i) {
  //   b->data[i] = (float) i;
  // }
  // print_matrix(matmul(a, b), 1);
  // run_rl(test);
  // init_demo_buffer(100, 34);
  // run_agent("DDPG_ACTOR_PICKNPLACE_NORM.model", 1, "DDPG_NORM_PICKNPLACE_NORM.norm", ENV_PICK_N_PLACE);
  // init_demo_connection();
  // build_demo_buffer(5016, 62);
  // print_experiences(b);
  return 0;
  #endif
}


#ifdef RUN_TEST
static int simple_test() {
  printf("TEST MODE\n\n");
  
  printf("Testing matrix operations...\n");
  test_results();
  printf("\n");
  printf("Testing data loading...\n");
  matrix_t* t;
  #ifndef C_AS_LIB
  t = load_data("dat/FM_dataset.dat");
  #else
  t = load_data("./src/robot_reinforcement_learning/C/FM_dataset.dat");
  #endif
  print_matrix(t, 0);
  printf("\n");
  printf("Testing with sample data, timing...\n");
  struct timeval start;
  timer_reset(&start);
  run_rl(test);
  float diff = timer_check(&start);
  printf("Test training took %.1f s\n", diff/1000);
  printf("\n");
  return 0;
}
#endif


#ifndef C_AS_LIB
int main(){
  return _main();
}
#endif

