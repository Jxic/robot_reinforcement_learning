#include "tests.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
matrix_t* mat_3_3_wiz_scalar();
matrix_t* mat_3_3_neg();
matrix_t* mat_3_3_wiz_factor();
matrix_t* mat_2_3();
matrix_t* mat_3_2();

char* matrix_test_elem_wise_add();
char* matrix_test_elem_wise_minus();
char* matrix_test_elem_wise_mult();
char* matrix_test_matmul();
char* matrix_test_transpose();
char* matrix_test_add_bias();
char* matrix_test_add_scalar();
char* matrix_test_equal();
char* matrix_test_neg();
char* matrix_test_mult_scalar();
char* matrix_test_mean();
char* matrix_test_copy_matrix();
char* matrix_test_slice_row_wise();
char* matrix_test_augment_space();
char* matrix_test_slice_col_wise();

char* test_all(){
  mu_run_test(matrix_test_equal);
  mu_run_test(matrix_test_elem_wise_add);
  mu_run_test(matrix_test_elem_wise_minus);
  mu_run_test(matrix_test_elem_wise_mult);
  mu_run_test(matrix_test_matmul);
  mu_run_test(matrix_test_transpose);
  mu_run_test(matrix_test_add_scalar);
  mu_run_test(matrix_test_add_bias);
  mu_run_test(matrix_test_neg);
  mu_run_test(matrix_test_mult_scalar);
  mu_run_test(matrix_test_mean);
  mu_run_test(matrix_test_copy_matrix);
  mu_run_test(matrix_test_slice_row_wise);
  mu_run_test(matrix_test_augment_space);
  mu_run_test(matrix_test_slice_col_wise);
  //printf("testing break\n");
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

char* matrix_test_slice_row_wise() {
  matrix_t* test_mat = mat_3_3();
  matrix_t* sliced = slice_row_wise(test_mat, 0, 2);
  mu_assert("[MATRIX_TEST_SLICE_ROW_WISE] wrong result slicing row-wise",
              equal(sliced, mat_2_3()));
  return 0;
}

char* matrix_test_slice_col_wise() {
  matrix_t* test_mat = mat_3_3();
  matrix_t* sliced = slice_col_wise(test_mat, 1, 3);
  mu_assert("[MATRIX_TEST_SLICE_COL_WISE] wrong result slicing col-wise",
              equal(sliced, mat_3_2()));
  return 0;
}

char* matrix_test_augment_space() {
  matrix_t* test_mat1 = mat_3_3();
  matrix_t* test_mat2 = mat_3_3();
  augment_space(test_mat2, 10, 10);
  mu_assert("[MATRIX_TEST_AUGMENT_SPACE] wrong result augmenting space",
            equal(test_mat1, test_mat2) && test_mat2->max_size > test_mat1->max_size);
  return 0;
}

char* matrix_test_copy_matrix() {
  matrix_t* test_mat = mat_4_3();
  matrix_t* copied = new_matrix(4, 3);
  copy_matrix(copied, test_mat);
  mu_assert("[MATRIX_TEST_COPY_MATRIX] wrong result copying",
            equal(test_mat, copied));
  return 0;
}

char* matrix_test_mean() {
  matrix_t* test_mat = mat_3_3();
  float answer = 5;
  mu_assert("[MATRIX_TEST_MEAN] wrong result averaging",
            mean(test_mat) == answer);
  return 0;
}

char* matrix_test_mult_scalar() {
  matrix_t* test_mat = mat_3_3();
  mult_scalar(test_mat, 3);
  mu_assert("[MATRIX_TEST_MULT_SCALAR] wrong result multiplying scalar",
            equal((mat_3_3_wiz_factor()), test_mat));
  return 0;
}

char* matrix_test_neg() {
  matrix_t* test_mat = mat_3_3();
  neg(test_mat);
  mu_assert("[MATRIX_TEST_NEG] wrong result finding negative result",
            equal(mat_3_3_neg(), test_mat));
  return 0;
}

char* matrix_test_add_scalar() {
  matrix_t* test_mat = mat_3_3();
  add_scalar(test_mat, 3);
  mu_assert("[MATRIX_TEST_ADD_SCALAR] wrong result adding scalar",
            equal(mat_3_3_wiz_scalar(), test_mat));
  return 0;
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
  matrix_t* res = new_matrix(3,3);
  matmul(mat_3_3(),mat_3_3(), res);
  mu_assert("[MATRIX_TEST_MATMUL] wrong result on matrix multiplication", 
            equal(mat_3_3_self_matmul(), res));

  return 0;
}

char* matrix_test_elem_wise_mult(){
  matrix_t* a = mat_3_3();
  elem_wise_mult(a, a);
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
  matrix_t* new_mat = new_matrix(3,3);
  for(int i = 0; i < 9; ++i) new_mat->data[i] = i + 1;
  return new_mat;
}

matrix_t* mat_3_3_self_add(){
  matrix_t* new_mat = new_matrix(3,3);
  for(int i = 0; i < 9; ++i) new_mat->data[i] = 2 * i + 2;
  return new_mat;
}

matrix_t* mat_3_3_self_minus(){
  matrix_t* new_mat = new_matrix(3,3);
  for(int i = 0; i < 9; ++i) new_mat->data[i] = 0;
  return new_mat;
}

matrix_t* mat_3_3_self_mult(){
  matrix_t* new_mat = new_matrix(3,3);
  for(int i = 0; i < 9; ++i) new_mat->data[i] = (i + 1) * (i + 1);
  return new_mat;
}

matrix_t* mat_3_3_self_matmul(){
  matrix_t* new_mat = new_matrix(3,3);
  float answer[] = {30,36,42,66,81,96,102,126,150};
  memcpy(new_mat->data, answer, 9*sizeof(float));
  return new_mat;
}

matrix_t* mat_3_4(){
  matrix_t* new_mat = new_matrix(3,4);
  for(int i = 0; i < 12; ++i) new_mat->data[i] = i + 1;
  return new_mat;
}

matrix_t* mat_4_3(){
  matrix_t* new_mat = new_matrix(4,3);
  float answer[] = {1,5,9,2,6,10,3,7,11,4,8,12};
  memcpy(new_mat->data, answer, 12*sizeof(float));
  return new_mat;
}

matrix_t* mat_1_3(){
  matrix_t* new_mat = new_matrix(1,3);
  float answer[] = {1,2,3};
  memcpy(new_mat->data, answer, 3*sizeof(float));
  return new_mat;
}

matrix_t* mat_3_3_wiz_bias(){
  matrix_t* new_mat = new_matrix(3,3);
  float answer[] = {2,4,6,5,7,9,8,10,12};
  memcpy(new_mat->data, answer, 9*sizeof(float));
  return new_mat;
}

matrix_t* mat_3_3_wiz_scalar() {
  matrix_t* new_mat = new_matrix(3,3);
  float answer[] = {4,5,6,7,8,9,10,11,12};
  memcpy(new_mat->data, answer, 9*sizeof(float));
  return new_mat;
}

matrix_t* mat_3_3_neg() {
  matrix_t* new_mat = new_matrix(3,3);
  float answer[] = {-1,-2,-3,-4,-5,-6,-7,-8,-9};
  memcpy(new_mat->data, answer, 9*sizeof(float));
  return new_mat;
}

matrix_t* mat_3_3_wiz_factor() {
  matrix_t* new_mat = new_matrix(3,3);
  float answer[] = {3,6,9,12,15,18,21,24,27};
  memcpy(new_mat->data, answer, 9*sizeof(float));
  return new_mat;
}

matrix_t* mat_2_3() {
  matrix_t* new_mat = new_matrix(2,3);
  float answer[] = {1,2,3,4,5,6};
  memcpy(new_mat->data, answer, 6*sizeof(float));
  return new_mat;
}

matrix_t* mat_3_2() {
  matrix_t* new_mat = new_matrix(3,2);
  float answer[] = {2,3,5,6,8,9};
  memcpy(new_mat->data, answer, 6*sizeof(float));
  return new_mat;
}
