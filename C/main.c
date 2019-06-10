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
#ifdef MKL
#include "mkl.h"
#endif

#ifdef RUN_TEST
static int simple_test();
#endif


int _main() {
  // preparation phase
  srand(SEED);

  #ifdef MKL
  #ifdef MULTI_MKL_THREAD
//  mkl_set_num_threads(4);
  #endif
  #endif

  #ifdef RUN_TEST  
  return simple_test();
  #else
  
  // matrix_t* a = new_matrix(3,4);
  // matrix_t* b = new_matrix(4,5);
  // for (int i = 0; i < 12; ++i) {
  //   a->data[i] = (double) i;
  // }
  // for (int i = 0; i < 20; ++i) {
  //   b->data[i] = (double) i;
  // }
  // print_matrix(matmul(a, b), 1);
  run_rl(ddpg);
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
  // test_results();
  printf("\n");
  printf("Testing data loading...\n");
  matrix_t* t;
  #ifndef C_AS_LIB
  t = load_data("FM_dataset.dat");
  #else
  t = load_data("./src/robot_reinforcement_learning/C/FM_dataset.dat");
  #endif
  print_matrix(t, 0);
  printf("\n");
  printf("Testing with sample data, timing...\n");
  struct timeval start;
  timer_reset(&start);
  run_rl(test);
  double diff = timer_check(&start);
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

