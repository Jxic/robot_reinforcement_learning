#include "matrix_op.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "macros.h"
#include <string.h>
#include <math.h>
#include "utils.h"
#ifdef MKL
#include "mkl.h"
#endif


void dummy(){
  printf("DUMMY FUNCTION");
}

int initialize(matrix_t* mat, initializer i) {
  switch (i) {
    case xavier:
      return xavier_init(mat, 1);
    case truncated_normal:
      return truncated_normal_init(mat);
    case zeros:
      return zero_init(mat);
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

int mult_bias(matrix_t* a, matrix_t* b) {
  assert(a->cols == b->cols);

  for (int i = 0; i < a->rows; ++i) {
    for (int j = 0; j < b->cols; ++j) {
      a->data[i*b->cols + j] *= b->data[j];
    }
  }

  return 1;
}

int add_scalar(matrix_t* a, double b) {
  assert(a->rows > 0 && a->cols > 0);
  for (int i = 0; i < a->rows*a->cols; ++i) a->data[i] += b;
  return 1;
}

int inverse(matrix_t* a) {
  assert(a->rows > 0 && a->cols > 0);
  for (int i = 0; i < a->rows*a->cols; ++i) {
    //assert(a->data[i]);
    a->data[i] = 1/a->data[i];
  } 
  return 1;
}

int square_root(matrix_t* a) {
  assert(a->rows > 0 && a->cols > 0);
  for (int i = 0; i < a->rows*a->cols; ++i) {
    assert(a->data[i]>=0);
    a->data[i] = sqrt(a->data[i]);
  } 
  return 1;
}

int neg(matrix_t* a) {
  assert(a->rows > 0 && a->cols > 0);
  for (int i = 0; i < a->rows*a->cols; ++i) a->data[i] = -a->data[i];
  return 1;
}

int mult_scalar(matrix_t* a, double b) {
  assert(a->rows > 0 && a->cols >0);
  #ifndef MKL
  for (int i = 0; i < a->rows*a->cols; ++i) a->data[i] *= b;
  #else
  MKL_INT n = a->rows*a->cols;
  MKL_INT inc = 1;
  cblas_dscal(n, b, a->data, inc);
  #endif
  return 1;
}

matrix_t* matmul(matrix_t* a, matrix_t* b) {
  assert(a->cols == b->rows);
  assert(a->rows * b->cols > 0);
  #ifdef MKL
  int m, n, p, i;
  double alpha, beta;
  m = a->rows;
  p = a->cols;
  n = b->cols;
  matrix_t* ret = new_matrix(m, n);
  // printf("m %d, n %d, p %d", m, n, p);
  alpha = 1.0;
  beta = 0.0;
  for (i = 0; i < m*n; i++) {
    ret->data[i] = 0.0;
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, p, alpha, a->data, p, b->data, n, beta, ret->data, n);
  return ret;
  #endif
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
  #ifndef MKL
  for (int i = 0; i < a->rows*a->cols; ++i) sum += a->data[i];
  #else
  sum = cblas_dasum(a->rows*a->cols, a->data, 1);
  #endif
  return sum / (double)(a->rows * a->cols);
}

int free_matrix(matrix_t* t) {
  #ifndef MKL
  free(t->data);
  #else
  MKL_free(t->data);
  #endif
  free(t);
  return 1;
}

int contains_nan(matrix_t* t) {
  for (int i = 0; i < t->rows*t->cols; ++i) {
    if (t->data[i] != t->data[i]) return 1;
  }
  return 0;
}

int augment_space(matrix_t* t, int rows, int cols) {
  assert(rows >= t->rows);
  assert(cols >= t->cols);
  t->max_size = rows * cols;
  #ifndef MKL
  t->data = realloc(t->data, rows*cols*sizeof(double));
  #else
  t->data = mkl_realloc(t->data, rows*cols*sizeof(double));
  #endif
  if (!t->data) {
    printf("[AUGMENT_SPACE] error reallocating memory");
    exit(1);
  }
  return 1;
}

int any_larger(matrix_t* t, double thres) {
  for (int i = 0; i < t->rows*t->cols; ++i) {
    if (t->data[i] > thres) {
      return 1;
    }
  }
  return 0;
}

int copy_matrix(matrix_t* dst, matrix_t* src) {
  assert(dst->max_size >= src->rows*src->cols);
  #ifndef MKL
  memcpy(dst->data, src->data, src->cols*src->rows*sizeof(double));
  #else
  MKL_INT n = src->rows*src->cols;
  MKL_INT incx = 1;
  MKL_INT incy = 1;
  cblas_dcopy(n, src->data, incx, dst->data, incy);
  #endif
  dst->rows = src->rows;
  dst->cols = src->cols;
  return 1;
}

int* shuffle_row_wise(matrix_t* t, int* pre_idx) {
  assert(t->rows > 0);
  assert(t->cols > 0);

  if (t->rows == 1) {
    return 0;
  }

  int size = t->rows;
  int* idx;
  if (!pre_idx) {
    idx = calloc(size, sizeof(int));
    for (int i = 0; i < size; ++i) idx[i] = i;
    for (int i = size - 1; i > 0; --i) {
      int j = rand() % (i + 1);
      int temp = idx[j];
      idx[j] = idx[i];
      idx[i] = temp;
    }
  } else {
    idx = pre_idx;
  }
  #ifndef MKL
  double* new_data = calloc(t->rows*t->cols, sizeof(double));
  #else
  double* new_data = MKL_calloc(t->rows*t->cols, sizeof(double), 64);
  #endif
  for (int i = 0; i < t->rows; ++i) memcpy(new_data+i*t->cols, t->data+idx[i]*t->cols, t->cols*sizeof(double));
  #ifndef MKL
  free(t->data);
  #else
  MKL_free(t->data);
  #endif
  t->data = new_data;
  return idx;
}

int print_matrix(matrix_t* t, int all) {
  printf("--------------------------------------\n");
  if (all) {
    for (int i = 0; i < t->rows; ++i) {
      for (int j = 0; j< t->cols; ++j) {
        printf(" %e ",t->data[i*t->cols+j]);
      }
      printf("\n");
    }
  } else {
    for (int i = 0; i < t->rows; i += t->rows/3) {
      for (int j = 0; j< t->cols; ++j) {
        printf(" %e ",t->data[i*t->cols+j]);
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
  #ifndef MKL
  new_m->data = calloc(rows*cols, sizeof(double));
  #else
  new_m->data = mkl_calloc(rows*cols, sizeof(double), 64);
  #endif
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
  if (!ret) {
    printf("[SLICE COL WISE] failed to create new matrix\n");
    exit(1);
  }
  int row_size = end - start;
  for (int i = 0; i < t->rows; ++i) {
    memcpy(ret->data+(i*ret->cols), t->data+(i*t->cols+start), row_size*sizeof(double));
  }
  return ret;
}

int xavier_init(matrix_t* a, double gain) {
  double low = -gain * sqrt((double)1 / (double)(a->rows));// + a->cols));
  double high = gain * sqrt((double)1 / (double)(a->rows));// + a-> cols));
  for (int i = 0; i < a->rows*a->cols; ++i) a->data[i] = rand_uniform(low, high);
  return 1;
}

int truncated_normal_init(matrix_t* t) {
  int size = t->cols*t->rows;
  matrix_t* samples = trunc_normal(size, 0.003, -0.003);
  for (int i = 0; i < t->rows*t->cols; ++i) t->data[i] = samples->data[i];
  free_matrix(samples);
  return 1;
}

int zero_init(matrix_t* t) {
  for (int i = 0; i < t->cols*t->rows; ++i) t->data[i] = 0;
  return 1;
}

matrix_t* normalize(matrix_t* t) {
  matrix_t* ret = new_matrix(2, t->cols);
  for (int i = 0; i < t->cols; ++i) {
    double max = t->data[i];
    double min = t->data[i];
    for (int j = 0; j < t->rows; ++j) {
      double curr_num = t->data[j*t->cols+i];
      if (curr_num > max) {
        max = curr_num;
      }
      if (curr_num < min) {
        min = curr_num;
      }
    }
    for (int j = 0; j < t->rows; ++j) {
      t->data[j*t->cols+i] = (t->data[j*t->cols+i] - min) / (max - min);
    }
    ret->data[i] = max;
    ret->data[t->cols+i] = min;
  }
  return ret;
}

int scale(matrix_t* x, matrix_t* min_max) {
  assert(x->cols == min_max->cols);
  matrix_t* max = slice_row_wise(min_max, 0, 1);
  matrix_t* min = slice_row_wise(min_max, 1, 2);
  matrix_t* diff = new_matrix(1, x->cols);
  copy_matrix(diff, max);
  elem_wise_minus(diff, min);

  mult_bias(x, diff);
  add_bias(x, min);
  free_matrix(max);
  free_matrix(min);
  free_matrix(diff);
  
  return 1;
}

matrix_t* concatenate(matrix_t* a, matrix_t* b, int axis) {
  // 1 horizontal, 0 vertical
  assert(axis == 0 || axis == 1);
  matrix_t* ret;
  if (axis) {
    assert(a->rows == b->rows);
    int rows = a->rows;
    int cols = a->cols + b->cols;
    ret = new_matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
      memcpy(ret->data+(i*cols), a->data+(i*a->cols), a->cols*sizeof(double));
      memcpy(ret->data+(i*cols)+a->cols, b->data+(i*b->cols), b->cols*sizeof(double));
    }
  } else {
    assert(a->cols == b->cols);
    int rows = a->rows + b->rows;
    int cols = a->cols;
    ret = new_matrix(rows, cols);
    //memcpy(ret->data, a->data, a->cols*a->rows*sizeof(double));
    //memcpy(ret->data+(a->rows*a->cols), b->data, b->cols*b->rows*sizeof(double));
    for (int i = 0; i < rows; ++i) memcpy(ret->data+(i*cols), a->data+(i*cols), cols*sizeof(double));
    for (int i = 0; i < rows; ++i) memcpy(ret->data+((i+a->cols)*cols), b->data+(i*cols), cols*sizeof(double));
  }
  return ret;
}

matrix_t* clone(matrix_t* a) {
  assert(a);
  matrix_t* ret = new_matrix(a->rows, a->cols);
  copy_matrix(ret, a);
  return ret;
}

int clip(matrix_t* a, double low, double high) {
  assert(high > low);
  for (int i = 0; i < a->rows*a->cols; ++i) {
    if (a->data[i] < low) {
      a->data[i] = low;
    }
    if (a->data[i] > high) {
      a->data[i] = high;
    }
  }
  return 1;
}
