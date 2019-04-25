#ifndef MATRIX_OP_H
#define MATRIX_OP_H

#include "data_structures.h"

typedef struct _matrix_t {
  double* data;
  int rows;
  int cols;
  int max_size;
} matrix_t;


typedef enum _initializer {
  xavier
} initializer;

void dummy();
int initialize(matrix_t* mat, initializer i);

int elem_wise_add(matrix_t* a, matrix_t* b);
int elem_wise_minus(matrix_t* a, matrix_t* b);
int elem_wise_mult(matrix_t* a, matrix_t* b);
int add_bias(matrix_t* a, matrix_t* b);
int add_scalar(matrix_t* a, double b);
int mult_scalar(matrix_t* a, double b);
int neg(matrix_t* a);

matrix_t* matmul(matrix_t* a, matrix_t* b);
matrix_t* transpose(matrix_t* a);

double mean(matrix_t* a);

int equal(matrix_t* a, matrix_t* b);

int xavier_init(matrix_t* a, double gain);

int free_matrix(matrix_t* t);
int copy_matrix(matrix_t* dst, matrix_t* src);
int augment_space(matrix_t* t, int rows, int cols);
matrix_t* new_matrix(int rows, int cols);
// matrix_t* shuffle_matrix_row_wise(matrix_t* t);
matrix_t* slice_row_wise(matrix_t* t, int start, int end);

#endif