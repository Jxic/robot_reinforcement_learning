#include "matrix_op.h"
#include <stdio.h>
#include <stdlib.h>
#include "tests.h"
#include "macros.h"
#include "utils.h"
#include "rl.h"
#include <time.h>
#include <string.h>
#ifdef MKL
#include "mkl.h"
#endif
// cmd + shift + p -> edit configuration

int _main() {
  // preparation phase
  srand(SEED);
  // char a[30];
  // memset(a, 0, 30);
  // for (int i = 0; i< 30; ++i) printf("%d ", (int)rand_uniform(0, 50));
  // matrix_t* ra = rand_normal(10);
  // print_matrix(ra, 1);
  // return 1;
  // test mode
  // matrix_t* m = new_matrix(2, 10);
  // initialize(m, xavier);
  // matrix_t* n = clone(m);
  // free_matrix(m);
  // for (int i = 0; i < 20; ++i) {
  //   printf("%f ", n->data[i]);
  // }
  // printf("\n");
  // exit(1);
  #ifdef RUN_TEST

  // matrix_t* a = new_matrix(3, 5);
  // initialize(a, xavier);
  // print_matrix(a,1 );
  // mult_scalar(a, 3);
  // print_matrix(a, 1);
  // augment_space(a, 3, 100);
  // matrix_t* b = new_matrix(10, 10);
  // copy_matrix(b, a);
  // print_matrix(b, 1);
  // matrix_t* c = new_matrix(5, 3);
  // initialize(c, xavier);
  // if (!equal(matmul(a, c), matmul(b, c))) printf("something's wrong!\n");
  // return 1;
  // matrix_t* rand_mat = new_matrix(3, 100);
  // initialize(rand_mat, truncated_normal);
  // print_matrix(rand_mat, 1);
  // // matrix_t* t_mat = trunc_normal(20, 2, -2);
  // // print_matrix(t_mat,1);
  // return 1;
  
  // printf("TEST MODE\n\n");
  // printf("Testing matrix operations...\n");
  // test_results();
  // printf("\n");
  // printf("Testing data loading...\n");
  // matrix_t* t;
  // #ifndef C_AS_LIB
  // t = load_data("FM_dataset.dat");
  // #else
  // t = load_data("./src/robot_reinforcement_learning/C/FM_dataset.dat");
  // #endif
  // print_matrix(t, 0);
  // printf("\n");
  // printf("Testing with sample data, timing...\n");
  clock_t start = clock(), diff;
  run_rl(test);
  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Test training took %ds %dms\n", msec/1000, msec%1000);
  printf("\n");
  return 0;
  // training mode
  #else
  run_rl(her);
  return 0;
  #endif
}

#ifndef C_AS_LIB
int main(){
  #ifdef MKL
  //mkl_set_num_threads(mkl_get_max_threads());
  #endif
  return _main();
}
#endif

