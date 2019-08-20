#ifndef MATRIX_OP_H
#define MATRIX_OP_H
#ifdef __cplusplus
extern "C" {
#endif
#include "macros.h"
typedef struct _matrix_t {
  float* data;
  int rows;
  int cols;
  int max_size;
} matrix_t;


typedef enum _initializer {
  xavier, truncated_normal, zeros, ones
} initializer;

void dummy();
int initialize(matrix_t* mat, initializer i);

int elem_wise_add(matrix_t* a, matrix_t* b);
int elem_wise_minus(matrix_t* a, matrix_t* b);
int elem_wise_mult(matrix_t* a, matrix_t* b);
int elem_wise_div(matrix_t* a, matrix_t* b, matrix_t* ret);
int add_bias(matrix_t* a, matrix_t* b);
int mult_bias(matrix_t* a, matrix_t* b);
int add_scalar(matrix_t* a, float b);
int mult_scalar(matrix_t* a, float b);
int neg(matrix_t* a);
int inverse(matrix_t* a);
int square_root(matrix_t* a, matrix_t* ret);
int square(matrix_t* a);
int matrix_exp(matrix_t* a);
int matrix_log(matrix_t* a);
matrix_t* matrix_sum(matrix_t* a, int axis);
matrix_t* matrix_row_argmax(matrix_t* a);

matrix_t** mat_mul_series(matrix_t* a, matrix_t* b, matrix_t* c, matrix_t* d, matrix_t* e, matrix_t* f);

int matmul(matrix_t* a, matrix_t* b, matrix_t* ret);
#ifdef GPU
matrix_t** matmul_gpu(matrix_t** ms, int count);
#endif
#ifdef MKL
int matmul_mkl(matrix_t* a, matrix_t* b, matrix_t* ret);
#endif
matrix_t* transpose(matrix_t* a);

float mean(matrix_t* a);

int equal(matrix_t* a, matrix_t* b);

int xavier_init(matrix_t* a, float gain);
int truncated_normal_init(matrix_t* a);
int zero_init(matrix_t* a);
int ones_init(matrix_t* a);

matrix_t* normalize(matrix_t* t);
int scale(matrix_t* x, matrix_t* min_max);
int free_matrix(matrix_t* t);
int contains_nan(matrix_t* t);
int print_matrix(matrix_t* t, int all);
int copy_matrix(matrix_t* dst, matrix_t* src);
int any_larger(matrix_t* t, float thres);
int augment_space(matrix_t* t, int rows, int cols);
int* shuffle_row_wise(matrix_t* t, int* idx);
matrix_t* new_matrix(int rows, int cols);
// matrix_t* shuffle_matrix_row_wise(matrix_t* t);
matrix_t* slice_row_wise(matrix_t* t, int start, int end);
matrix_t* slice_col_wise(matrix_t* t, int start, int end);
matrix_t* concatenate(matrix_t* a, matrix_t* b, int axis);
matrix_t* matrix_clone(matrix_t* a);
matrix_t* one_hot_encoding(matrix_t* a, int size);
int clip(matrix_t* a, float low, float high);


#ifdef __cplusplus
}
#endif
#endif
