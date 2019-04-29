#include "matrix_op.h"
#include <stdio.h>
#include <stdlib.h>
#include "tests.h"
#include "macros.h"
#include "utils.h"
#include "rl.h"
#include <time.h>
// cmd + shift + p -> edit configuration

int _main() {
  // preparation phase
  srand(SEED);
  // test mode
  #ifdef RUN_TEST
  printf("TEST MODE\n\n");
  printf("Testing matrix operations...\n");
  test_results();
  printf("\n");
  printf("Testing model constructions...\n");
  init_rl_model(0);
  printf("\n");
  printf("Testing data loading...\n");
  matrix_t* t;
  #ifndef C_AS_LIB
  t = load_data("FM_dataset.dat");
  #else
  t = load_data("./src/C/FM_dataset.dat");
  #endif
  print_matrix(t, 0);
  printf("\n");
  printf("Testing with sample data, timing...\n");
  clock_t start = clock(), diff;
  model* m = init_rl_model(0);
  run_model(m);
  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Test training took %ds %dms\n", msec/1000, msec%1000);
  printf("\n");
  return 0;
  // training mode
  #else
  return 0;
  #endif
}

#ifndef C_AS_LIB
int main(){
  return _main();
}
#endif

