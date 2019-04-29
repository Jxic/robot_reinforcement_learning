#include "matrix_op.h"
#include <stdio.h>
#include <stdlib.h>
#include "tests.h"
#include "macros.h"
#include "utils.h"
#include "rl.h"
#include <time.h>
// cmd + shift + p -> edit configuration

int main() {
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
  matrix_t* loaded_data = load_data("FM_dataset.dat");
  print_matrix(loaded_data, 0);
  printf("\n");
  printf("Testing with sample data, timing...\n");
  clock_t start = clock(), diff;
  test_run();
  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Test training takes %ds %dms\n", msec/1000, msec%1000);
  printf("\n");
  return 0;
  // training mode
  #else
  return 0;
  #endif
}
