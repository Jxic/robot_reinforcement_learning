#include "matrix_op.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "macros.h"
void dummy(){
  printf("DUMMY FUNCTION");
}

int equal(matrix_t* a, matrix_t* b){
  if (a->rows != b->rows || a->cols != b->cols) {
    #ifndef RUN_TEST
    printf("[MATRICES NOT EQUAL] size is different");
    #endif
    return 0;
  }
  if (memcmp(a->data, b->data, (a->rows*a->cols)*sizeof(double))) {
    #ifndef RUN_TEST
    printf("[MATRICES NOT EQUAL] data is different");
    #endif
    return 0;
  }
  return 1;
}

int elem_wise_add(matrix_t* a, matrix_t* b){
  assert(a->rows == b->rows);
  assert(b->cols == b->cols);

  int size = a->rows*a->cols;
  for (int i = 0; i < size; ++i) {
    a->data[i] += b->data[i];
  }
  
  return 0;
}

int elem_wise_minus(matrix_t* a, matrix_t* b){
  assert(a->rows == b->rows);
  assert(b->cols == b->cols);

  int size = a->rows*a->cols;
  for (int i = 0; i < size; ++i) {
    a->data[i] -= b->data[i];
  }
  
  return 0;
}

int elem_wise_mult(matrix_t* a, matrix_t* b){
  assert(a->rows == b->rows);
  assert(b->cols == b->cols);

  int size = a->rows*a->cols;
  for (int i = 0; i < size; ++i) {
    a->data[i] *= b->data[i];
  }
  
  return 0;
}

matrix_t* matmul(matrix_t* a, matrix_t* b){
  assert(a->cols == b->rows);
  assert(a->rows * b->cols > 0);
  matrix_t* new_mat = (matrix_t*)malloc(sizeof(matrix_t));
  if (!new_mat) {
    printf("[MATMUL] malloc failure");
    exit(1);
  }
  new_mat->rows = a->rows;
  new_mat->cols = b->cols;
  new_mat->data = calloc(new_mat->rows * new_mat->cols, sizeof(double));
  if (!new_mat->data) {
    printf("[MATMUL] calloc failure");
    exit(1);
  }
  for (int i = 0; i < new_mat->rows; ++i) {
    for (int k = 0; k < a->cols; ++k) {
      for (int j = 0; j < new_mat->cols; ++j) {
        new_mat->data[i*new_mat->cols+j] += a->data[i*a->cols+k] * b->data[k*b->cols+j];
      }
    }
  }
  return new_mat;
}

matrix_t* transpose(matrix_t* a) {
  matrix_t* new_mat = (matrix_t*)malloc(sizeof(matrix_t));
  if (!new_mat) {
    printf("[TRANSPOSE] malloc failure");
    exit(1);
  }
  new_mat->cols = a->rows;
  new_mat->rows = a->cols;
  new_mat->data = calloc(a->rows*a->cols, sizeof(double));
  if (!new_mat) {
    printf("[TRANSPOSE] calloc failure");
    exit(1);
  }
  for (int i = 0; i < new_mat->rows; ++i) {
    for (int j = 0; j < new_mat->cols; ++j) {
      new_mat[i*new_mat->cols+j] = a[j*a->cols+i];
    }
  }
  return new_mat;
}

