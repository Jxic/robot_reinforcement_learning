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
  printf("Test training takes %d msecs\n", msec);
  printf("\n");
  // matrix_t* test_mat = new_matrix(30,6);
  // for (int i = 0; i < 180; ++i) test_mat->data[i] = i + 1;
  // print_matrix(test_mat, 1);
  // shuffle_row_wise(test_mat);
  // matrix_t* f_3 = slice_col_wise(test_mat, 0, 3);
  // matrix_t* b_3 = slice_col_wise(test_mat, 3, 6);
  
  // print_matrix(f_3, 1);
  // print_matrix(b_3, 1);
  return 0;
  // training mode
  #else
  return 0;
  #endif
}
