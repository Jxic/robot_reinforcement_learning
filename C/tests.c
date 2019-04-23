#include "tests.h"
#include <stdio.h>
#include <stdlib.h>
#include "data_structures.h"
#include "matrix_op.h"

int tests_run = 0;

matrix_t* mat_3_3();
matrix_t* mat_3_3_self_add();
matrix_t* mat_3_3_3();
matrix_t* mat_2_dim_add_3_dim();

char* matrix_test_elem_wise_add();
char* matrix_test_elem_wise_minus();
char* matrix_test_elem_wise_mult();
char* matrix_test_elem_wise_div();
char* matrix_test_equal();

char* test_all(){
  mu_run_test(matrix_test_equal);
  mu_run_test(matrix_test_elem_wise_add);
  return 0;
}

void test_results(){
  char* result = test_all();
  if(result != 0){
    printf("%s\n", result);
  } else {
    printf("ALL TESTS PASSED (in theory...)\n");
  }
  printf("TESTS RUN: %d\n", tests_run);
  if(result != 0) {
    exit(0);
  }
}

char* matrix_test_elem_wise_add(){
  mu_assert("[MATRIX_TEST_ELEM_WISE_ADD] wrong result on same dimension add", 
            equal(mat_3_3_self_add(), elem_wise_add(mat_3_3(), mat_3_3())));
  mu_assert("[MATRIX_TEST_ELEM_WISE_ADD] wrong result on different dimension add",
            equal(mat_2_dim_add_3_dim(), elem_wise_add(mat_3_3(), mat_3_3_3())));
  mu_assert("[MATRIX_TEST_ELEM_WISE_ADD] wrong result on different dimension add",
            equal(mat_2_dim_add_3_dim(), elem_wise_add(mat_3_3_3(), mat_3_3())));
  // double* data_a = mat_2_dim_add_3_dim()->data;
  // double* data_b = elem_wise_add(mat_3_3(), mat_3_3_3())->data;
  // for(int i = 0; i < 27; i++) {
  //   printf("%f, %f\n", data_a[i], data_b[i]);
  // }
  return 0;
}

char* matrix_test_equal(){
  mu_assert("[MATRIX_TEST_EQUAL] matrix should be equal", equal(mat_3_3(), mat_3_3()));
  mu_assert("[MATRIX_TEST_EQUAL] matrix should not be equal", !equal(mat_3_3(), mat_3_3_self_add()));
  return 0;
}

matrix_t* mat_3_3(){
  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* data = calloc(9, sizeof(double));
  int* dimension = calloc(3, sizeof(int));
  for(int i = 0; i < 9; ++i) data[i] = i + 1;
  dimension[0] = 2;
  dimension[1] = 3;
  dimension[2] = 3;
  new_mat->data = data;
  new_mat->dimensions = dimension;
  new_mat->size = 9;
  return new_mat;
}

matrix_t* mat_3_3_self_add(){
  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* data = calloc(9, sizeof(double));
  int* dimension = calloc(3, sizeof(int));
  for(int i = 0; i < 9; ++i) data[i] = 2 * i + 2;
  dimension[0] = 2;
  dimension[1] = 3;
  dimension[2] = 3;
  new_mat->data = data;
  new_mat->dimensions = dimension;
  new_mat->size = 9;
  return new_mat;
}

matrix_t* mat_3_3_3(){
  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* data = calloc(27, sizeof(double));
  int* dimension = calloc(4, sizeof(int));
  for(int i = 0; i < 27; ++i) data[i] = i + 1;
  dimension[0] = 4;
  dimension[1] = 3;
  dimension[2] = 3;
  dimension[3] = 3;
  new_mat->data = data;
  new_mat->dimensions = dimension;
  new_mat->size = 27;
  return new_mat;
}

matrix_t* mat_2_dim_add_3_dim(){
  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* data = calloc(27, sizeof(double));
  int* dimension = calloc(4, sizeof(int));
  for(int i = 0; i < 27; ++i) data[i] = i + 1 + ((i + 1) % 9);
  data[8] += 9;
  data[17] += 9;
  data[26] += 9;
  dimension[0] = 4;
  dimension[1] = 3;
  dimension[2] = 3;
  dimension[3] = 3;
  new_mat->data = data;
  new_mat->dimensions = dimension;
  new_mat->size = 27;
  return new_mat;
}
