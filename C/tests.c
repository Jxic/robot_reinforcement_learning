#include "tests.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_structures.h"
#include "matrix_op.h"

int tests_run = 0;

matrix_t* mat_3_3();
matrix_t* mat_3_3_self_add();
matrix_t* mat_3_3_self_minus();
matrix_t* mat_3_3_self_mult();
matrix_t* mat_3_3_self_matmul();
matrix_t* mat_4_3();
matrix_t* mat_3_4();
matrix_t* mat_1_3();
matrix_t* mat_3_3_wiz_bias();

char* matrix_test_elem_wise_add();
char* matrix_test_elem_wise_minus();
char* matrix_test_elem_wise_mult();
char* matrix_test_matmul();
char* matrix_test_transpose();
char* matrix_test_add_bias();
char* matrix_test_equal();

char* test_all(){
  mu_run_test(matrix_test_equal);
  mu_run_test(matrix_test_elem_wise_add);
  mu_run_test(matrix_test_elem_wise_minus);
  mu_run_test(matrix_test_elem_wise_mult);
  mu_run_test(matrix_test_matmul);
  mu_run_test(matrix_test_transpose);
  mu_run_test(matrix_test_add_bias);
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

char* matrix_test_add_bias() {
  matrix_t* test_mat = mat_3_3();
  add_bias(test_mat, mat_1_3());
  mu_assert("[MATRIX_TEST_ADD_BIAS] wrong result adding bias",
              equal(mat_3_3_wiz_bias(), test_mat));
  return 0;
}

char* matrix_test_transpose(){
  mu_assert("[MATRIX_TEST_MATMUL] wrong result on matrix transpose", 
            equal(transpose(mat_3_4()), mat_4_3()));
  return 0;
}

char* matrix_test_matmul(){
  matrix_t* res = matmul(mat_3_3(),mat_3_3());
  mu_assert("[MATRIX_TEST_MATMUL] wrong result on matrix multiplication", 
            equal(mat_3_3_self_matmul(), res));
  return 0;
}

char* matrix_test_elem_wise_mult(){
  matrix_t* a = mat_3_3();
  elem_wise_mult(a, mat_3_3());
  mu_assert("[MATRIX_TEST_ELEM_WISE_MULT] wrong result on same dimension multiplication", 
            equal(mat_3_3_self_mult(), a));
  return 0;
}

char* matrix_test_elem_wise_minus(){
  matrix_t* a = mat_3_3();
  elem_wise_minus(a, mat_3_3());
  mu_assert("[MATRIX_TEST_ELEM_WISE_MINUS] wrong result on same dimension minus", 
            equal(mat_3_3_self_minus(), a));
  return 0;
}

char* matrix_test_elem_wise_add(){
  matrix_t* a = mat_3_3();
  elem_wise_add(a, mat_3_3());
  mu_assert("[MATRIX_TEST_ELEM_WISE_ADD] wrong result on same dimension add", 
            equal(mat_3_3_self_add(), a));
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
  for(int i = 0; i < 9; ++i) data[i] = i + 1;
  new_mat->data = data;
  new_mat->rows = 3;
  new_mat->cols = 3;
  return new_mat;
}

matrix_t* mat_3_3_self_add(){
  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* data = calloc(9, sizeof(double));
  for(int i = 0; i < 9; ++i) data[i] = 2 * i + 2;
  new_mat->data = data;
  new_mat->rows = 3;
  new_mat->cols = 3;
  return new_mat;
}

matrix_t* mat_3_3_self_minus(){
  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* data = calloc(9, sizeof(double));
  for(int i = 0; i < 9; ++i) data[i] = 0;
  new_mat->data = data;
  new_mat->rows = 3;
  new_mat->cols = 3;
  return new_mat;
}

matrix_t* mat_3_3_self_mult(){
  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* data = calloc(9, sizeof(double));
  for(int i = 0; i < 9; ++i) data[i] = (i + 1) * (i + 1);
  new_mat->data = data;
  new_mat->rows = 3;
  new_mat->cols = 3;
  return new_mat;
}

matrix_t* mat_3_3_self_matmul(){
  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* data = calloc(9, sizeof(double));
  double answer[] = {30,36,42,66,81,96,102,126,150};
  memcpy(data, answer, 9*sizeof(double));
  new_mat->data = data;
  new_mat->rows = 3;
  new_mat->cols = 3;
  return new_mat;
}

matrix_t* mat_3_4(){
  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* data = calloc(12, sizeof(double));
  for(int i = 0; i < 12; ++i) data[i] = i + 1;
  new_mat->data = data;
  new_mat->rows = 3;
  new_mat->cols = 4;
  return new_mat;
}

matrix_t* mat_4_3(){
  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* data = calloc(12, sizeof(double));
  double answer[] = {1,5,9,2,6,10,3,7,11,4,8,12};
  memcpy(data, answer, 12*sizeof(double));
  new_mat->data = data;
  new_mat->rows = 4;
  new_mat->cols = 3;
  return new_mat;
}

matrix_t* mat_1_3(){
  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* data = calloc(3, sizeof(double));
  double answer[] = {1,2,3};
  memcpy(data, answer, 3*sizeof(double));
  new_mat->data = data;
  new_mat->rows = 1;
  new_mat->cols = 3;
  return new_mat;
}

matrix_t* mat_3_3_wiz_bias(){
  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* data = calloc(9, sizeof(double));
  double answer[] = {2,4,6,5,7,9,8,10,12};
  memcpy(data, answer, 9*sizeof(double));
  new_mat->data = data;
  new_mat->rows = 3;
  new_mat->cols = 3;
  return new_mat;
}
