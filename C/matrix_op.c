#include "matrix_op.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "macros.h"
#include <string.h>
#include <math.h>
#include "utils.h"

void dummy(){
  printf("DUMMY FUNCTION");
}

int initialize(matrix_t* mat, initializer i) {
  switch (i) {
    case xavier:
      return xavier_init(mat, 1);
    default:
      printf("[INITIALIZE] unrecognized initializer type");
      exit(1);
  }
}

int equal(matrix_t* a, matrix_t* b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    #ifdef RUN_TEST
    printf("[MATRICES NOT EQUAL] size is different\n");
    #endif
    return 0;
  }
  if (memcmp(a->data, b->data, (a->rows*a->cols)*sizeof(double))) {
    #ifdef RUN_TEST
    printf("[MATRICES NOT EQUAL] data is different\n");
    #endif
    return 0;
  }
  return 1;
}

int elem_wise_add(matrix_t* a, matrix_t* b) {
  assert(a->rows == b->rows);
  assert(b->cols == b->cols);

  int size = a->rows*a->cols;
  for (int i = 0; i < size; ++i) {
    a->data[i] += b->data[i];
  }
  
  return 1;
}

int elem_wise_minus(matrix_t* a, matrix_t* b) {
  assert(a->rows == b->rows);
  assert(b->cols == b->cols);

  int size = a->rows*a->cols;
  for (int i = 0; i < size; ++i) {
    a->data[i] -= b->data[i];
  }
  
  return 1;
}

int elem_wise_mult(matrix_t* a, matrix_t* b) {
  assert(a->rows == b->rows);
  assert(b->cols == b->cols);

  int size = a->rows*a->cols;
  for (int i = 0; i < size; ++i) {
    a->data[i] *= b->data[i];
  }
  
  return 1;
}

int add_bias(matrix_t* a, matrix_t* b) {
  assert(a->cols == b->cols);

  for (int i = 0; i < a->rows; ++i) {
    for (int j = 0; j < b->cols; ++j) {
      a->data[i*b->cols + j] += b->data[j];
    }
  }

  return 1;
}

int add_scalar(matrix_t* a, double b) {
  assert(a->rows > 0 && a->cols >0);
  for (int i = 0; i < a->rows*a->cols; ++i) a->data[i] += b;
  return 1;
}

int neg(matrix_t* a) {
  assert(a->rows > 0 && a->cols > 0);
  for (int i = 0; i < a->rows*a->cols; ++i) a->data[i] = -a->data[i];
  return 1;
}

int mult_scalar(matrix_t* a, double b) {
  assert(a->rows > 0 && a->cols >0);
  for (int i = 0; i < a->rows*a->cols; ++i) a->data[i] *= b;
  return 1;
}

matrix_t* matmul(matrix_t* a, matrix_t* b) {
  assert(a->cols == b->rows);
  assert(a->rows * b->cols > 0);
  matrix_t* new_mat = new_matrix(a->rows, b->cols);
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
  matrix_t* new_mat = new_matrix(a->cols, a->rows);
  for (int i = 0; i < new_mat->rows; ++i) {
    for (int j = 0; j < new_mat->cols; ++j) {
      new_mat->data[i*new_mat->cols+j] = a->data[j*a->cols+i];
    }
  }
  return new_mat;
}

double mean(matrix_t* a) {
  assert(a->rows > 0 && a->cols >0);
  double sum = 0;
  for (int i = 0; i < a->rows*a->cols; ++i) sum += a->data[i];
  return sum / (a->rows * a->cols);
}

int free_matrix(matrix_t* t) {
  free(t->data);
  free(t);
  return 1;
}

int augment_space(matrix_t* t, int rows, int cols) {
  assert(rows >= t->rows);
  assert(cols >= t->cols);
  t->max_size = rows * cols;
  t->data = realloc(t->data, rows*cols*sizeof(double));
  if (!t->data) {
    printf("[AUGMENT_SPACE] error reallocating memory");
    exit(1);
  }
  return 1;
}

int copy_matrix(matrix_t* dst, matrix_t* src) {
  assert(dst->max_size >= src->max_size);
  memcpy(dst->data, src->data, src->cols*src->rows*sizeof(double));
  dst->rows = src->rows;
  dst->cols = dst->cols;
  return 1;
}

int print_matrix(matrix_t* t, int all) {
  printf("--------------------------------------\n");
  if (all) {
    for (int i = 0; i < t->rows; ++i) {
      for (int j = 0; j< t->cols; ++j) {
        printf(" %lf ",t->data[i*t->cols+j]);
      }
      printf("\n");
    }
  } else {
    for (int i = 0; i < t->rows; i += t->rows/3) {
      for (int j = 0; j< t->cols; ++j) {
        printf(" %lf ",t->data[i*t->cols+j]);
      }
      printf("\n .......\n");
    } 
  }
  printf("--------------------------------------\n");
  printf(" rows: %d\n cols: %d\n max_size: %d\n", t->rows, t->cols, t->max_size);
  printf("--------------------------------------\n");
  return 1;
}

matrix_t* new_matrix(int rows, int cols) {
  matrix_t* new_m = malloc(sizeof(matrix_t));
  new_m->data = calloc(rows*cols, sizeof(double));
  assert(new_m && new_m->data);
  new_m->rows = rows;
  new_m->cols = cols;
  new_m->max_size = rows * cols;
  return new_m;
}

// matrix_t* shuffle_matrix_row_wise(matrix_t* t) {
//   int rows = t->rows;
//   int cols = t->cols;
  
// }

matrix_t* slice_row_wise(matrix_t* t, int start, int end) {
  assert(start < end);
  assert(start >= 0);
  assert(end <= t->rows);
  matrix_t* ret = new_matrix(end-start, t->cols);
  int t_start = start * t->cols;
  int t_size = (end - start) * t->cols;
  memcpy(ret->data, t->data+t_start, t_size*sizeof(double));
  return ret;
}

matrix_t* slice_col_wise(matrix_t* t, int start, int end) {
  assert(start < end);
  assert(start >= 0);
  assert(end <= t->cols);
  matrix_t* ret = new_matrix(t->rows, end-start);
  int row_size = end - start;
  for (int i = 0; i < t->rows; ++i) {
    memcpy(ret->data+(i*ret->cols), t->data+(i*t->cols+start), row_size*sizeof(double));
  }
  return ret;
}

int xavier_init(matrix_t* a, double gain) {
  double low = -gain * sqrt((double)6 / (double)(a->rows + a->cols));
  double high = gain * sqrt((double)6 / (double)(a->rows + a-> cols));
  for (int i = 0; i < a->rows*a->cols; ++i) a->data[i] = rand_uniform(low, high);
  return 1;
}

