#include "matrix_op.h"
#include <stdio.h>
#include <stdlib.h>
#include "tests.h"
#include "macros.h"
#include "utils.h"
#include "rl.h"
#include <time.h>
#include <string.h>
// cmd + shift + p -> edit configuration

int _main() {
  // preparation phase
  srand(SEED);
  // char a[30];
  // memset(a, 0, 30);
  // for (int i = 0; i< 30; ++i) printf("%f ", rand_uniform(-3.14, 3.14));
  // matrix_t* ra = rand_normal(10);
  // print_matrix(ra, 1);
  // return 1;
  // test mode
  #ifdef RUN_TEST
  printf("TEST MODE\n\n");
  printf("Testing matrix operations...\n");
  test_results();
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
  clock_t start = clock(), diff;
  run_rl(test);
  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Test training took %ds %dms\n", msec/1000, msec%1000);
  printf("\n");
  return 0;
  // training mode
  #else
  run_rl(ddpg);
  return 0;
  #endif
}

#ifndef C_AS_LIB
int main(){
  return _main();
}
#endif

