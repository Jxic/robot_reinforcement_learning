#include "matrix_op.h"
#include <stdio.h>
#include <stdlib.h>
#include "tests.h"
#include "macros.h"
#include "utils.h"
#include "rl.h"
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
  return 0;
  // training mode
  #else
  return 0;
  #endif
}
