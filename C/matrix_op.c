#include "matrix_op.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "macros.h"
#include <string.h>
#include <math.h>
#include <time.h>
#include "utils.h"
#ifdef MKL
#include "mkl.h"
#include "mkl_vml.h"
#endif

#ifdef GPU
#include "cuda_runtime.h"
#include "cublas_v2.h"
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
    case ones:
      return ones_init(mat);
    default:
      printf("[INITIALIZE] unrecognized initializer type");
      exit(1);
  }
}

int equal(matrix_t* a, matrix_t* b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    #ifndef RUN_TEST
    printf("[MATRICES NOT EQUAL] size is different\n");
    #endif
    return 0;
  }
  if (memcmp(a->data, b->data, (a->rows*a->cols)*sizeof(float))) {
    #ifndef RUN_TEST
    printf("[MATRICES NOT EQUAL] data is different\n");
    #endif
    return 0;
  }
  return 1;
}

int elem_wise_add(matrix_t* a, matrix_t* b) {
  assert(a->rows == b->rows);
  assert(b->cols == b->cols);
  #ifdef MKL
  int n = a->rows*a->cols;
  int alpha = 1;
  int incx = 1;
  int incy = 1;
  cblas_saxpy(n, alpha, b->data, incx, a->data, incy);
  return 1;
  #endif
  int size = a->rows*a->cols;
  for (int i = 0; i < size; ++i) {
    a->data[i] += b->data[i];
  }
  
  return 1;
}

int elem_wise_minus(matrix_t* a, matrix_t* b) {
  assert(a->rows == b->rows);
  assert(b->cols == b->cols);
  #ifdef MKL
  int n = a->rows*a->cols;
  int alpha = -1;
  int incx = 1;
  int incy = 1;
  cblas_saxpy(n, alpha, b->data, incx, a->data, incy);
  return 1;
  #endif
  int size = a->rows*a->cols;
  for (int i = 0; i < size; ++i) {
    a->data[i] -= b->data[i];
  }
  
  return 1;
}

int elem_wise_mult(matrix_t* a, matrix_t* b) {
  assert(a->rows == b->rows);
  assert(b->cols == b->cols);
  #ifdef MKL
  vsMul(a->rows*a->cols, a->data, b->data, a->data);
  // free(a->data);
  // a->data = ret;
  return 1;
  #endif

  int size = a->rows*a->cols;
  for (int i = 0; i < size; ++i) {
    a->data[i] *= b->data[i];
  }
  
  return 1;
}

int elem_wise_div(matrix_t* a, matrix_t* b, matrix_t* ret) {
  assert(a->rows == b->rows);
  assert(a->cols == b->cols);
  assert(ret->cols == b->cols);
  assert(a->rows == ret->rows);
  #ifdef MKL
  vsDiv(a->rows*a->cols, a->data, b->data, ret->data);
  // free(a->data);
  // a->data = ret;
  return 1;
  #endif

  int size = a->rows*a->cols;
  for (int i = 0; i < size; ++i) {
    ret->data[i] = a->data[i] / b->data[i];
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

int add_scalar(matrix_t* a, float b) {
  assert(a->rows > 0 && a->cols > 0);
  #ifdef MKL
  vsLinearFrac( a->rows*a->cols, a->data, a->data, 1, b, 0, 1, a->data );
  return 1;
  #endif
  for (int i = 0; i < a->rows*a->cols; ++i) a->data[i] += b;
  return 1;
}

int inverse(matrix_t* a) {
  assert(a->rows > 0 && a->cols > 0);
  // #ifdef MKL
  // vdInv(a->rows*a->cols, a->data, a->data);
  // return 1;
  // #endif
  for (int i = 0; i < a->rows*a->cols; ++i) {
    //assert(a->data[i]);
    a->data[i] = 1/a->data[i];
  } 
  return 1;
}

int square_root(matrix_t* a, matrix_t* ret) {
  assert(a->rows > 0 && a->cols > 0);
  assert(ret->rows == a->rows && ret->cols == a->cols);
  #ifdef MKL
  vsSqrt(a->rows*a->cols, a->data, ret->data);
  return 1;
  #endif
  for (int i = 0; i < a->rows*a->cols; ++i) {
    assert(a->data[i]>=0);
    ret->data[i] = sqrt(a->data[i]);
  } 
  return 1;
}

int square(matrix_t* a) {
  assert(a->rows > 0 && a->cols > 0);
  #ifdef MKL
  vsSqr(a->rows*a->cols, a->data, a->data);
  return 1;
  #endif
  for (int i = 0; i < a->rows*a->cols; ++i) {
    //assert(a->data[i]>=0);
    a->data[i] = pow(a->data[i],2);
  } 
  return 1;
}

int matrix_exp(matrix_t* a) {
  assert(a->rows > 0 && a->cols > 0);
  for (int i = 0; i < a->rows*a->cols; ++i) a->data[i] = exp(a->data[i]);
  return 1;

}

int neg(matrix_t* a) {
  assert(a->rows > 0 && a->cols > 0);
  // #ifdef MKL
  // mult_scalar(a, -1);
  // #endif
  for (int i = 0; i < a->rows*a->cols; ++i) a->data[i] = -a->data[i];
  return 1;
}

int mult_scalar(matrix_t* a, float b) {
  assert(a->rows > 0 && a->cols >0);
  #ifdef MKL
  MKL_INT n = a->rows*a->cols;
  MKL_INT inc = 1;
  cblas_sscal(n, b, a->data, inc);
  return 1;
  #endif
  for (int i = 0; i < a->rows*a->cols; ++i) a->data[i] *= b;
  return 1;
}

#ifdef GPU
matrix_t** matmul_gpu(matrix_t** ms, int count) {
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  matrix_t** rets = calloc(count/2, sizeof(matrix_t*));
  matrix_t *a, *b;
  for (int i = 0; i < count/2; ++i) {
    a = ms[i*2];
    b = ms[i*2+1];
    int row_a = a->rows;
    int col_a = a->cols;
    int row_b = b->rows;
    int col_b = b->cols;
    int row_c = row_a;
    int col_c = col_b;

    matrix_t* ret = new_matrix(row_c, col_c);
    float* dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, row_a * col_a * sizeof(float));
    cudaMalloc((void**)&dev_b, row_b * col_b * sizeof(float));
    cudaMalloc((void**)&dev_c, row_c * col_c * sizeof(float));

    //int i, j;
    cublasSetMatrix(row_a, col_a, sizeof(float), a->data, row_a, dev_a, row_a);
    cublasSetMatrix(row_b, col_b, sizeof(float), b->data, row_b, dev_b, row_b);

  

    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to create gpu task handle\n");
    }

    float alpha = 1;
    float beta = 0;

    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, col_b, row_a, col_a, &alpha, dev_b, col_b, dev_a, col_a, &beta, dev_c, col_b);

    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed GEMM\n");
    }

    cublasGetMatrix(row_c, col_c, sizeof(float), dev_c, row_c, ret->data, row_c);
    rets[i] = ret;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
  }

  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("Failed destroying handle\n");
  }

  
  return rets;
}


matrix_t** mat_mul_series(matrix_t* a, matrix_t* b, matrix_t* c, matrix_t* d, matrix_t* e, matrix_t* f) {
  matrix_t** m_list = calloc(6, sizeof(matrix_t*));
  m_list[0] = a;
  m_list[1] = b;
  m_list[2] = c;
  m_list[3] = d;
  m_list[4] = e;
  m_list[5] = f;
  matrix_t** gpu_ret = matmul_gpu(m_list, 6);
  return gpu_ret;
}
#endif

#ifdef MKL
int matmul_mkl(matrix_t* a, matrix_t* b, matrix_t* ret) {
  int m, n, p;
  float alpha, beta;
  m = a->rows;
  p = a->cols;
  n = b->cols;
  // printf("m %d, n %d, p %d", m, n, p);
  alpha = 1.0;
  beta = 0.0;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, p, alpha, a->data, p, b->data, n, beta, ret->data, n);
  return 1;
}
#endif

int matmul(matrix_t* a, matrix_t* b, matrix_t* ret) {
  if (a->cols != b->rows) {
    printf("matmul a (%d %d) b (%d %d)\n", a->rows, a->cols, b->rows, b->cols);
    exit(1);
  }
  assert(a->cols == b->rows);
  assert(a->rows * b->cols > 0);
  assert(ret->max_size >= a->rows*b->cols);

  // #ifdef GPU
  // if (a->rows*a->cols >= 40000) {
  //   matrix_t** m_list = calloc(2, sizeof(matrix_t*));
  //   m_list[0] = a;
  //   m_list[1] = b;
  //   matrix_t** gpu_ret =  matmul_gpu(m_list, 2);
  //   matrix_t* gpu_ret0 = gpu_ret[0];
  //   free(gpu_ret);

  //   return gpu_ret0;
  // }
  // #endif
  
  #ifdef MKL
  ret->rows = a->rows;
  ret->cols = b->cols;

  return matmul_mkl(a, b, ret);
  #endif
  matrix_t* new_mat = new_matrix(a->rows, b->cols);
  for (int i = 0; i < new_mat->rows; ++i) {
    for (int k = 0; k < a->cols; ++k) {
      for (int j = 0; j < new_mat->cols; ++j) {
        new_mat->data[i*new_mat->cols+j] += a->data[i*a->cols+k] * b->data[k*b->cols+j];
      }
    }
  }
  copy_matrix(ret, new_mat);
  free_matrix(new_mat);
  return 1;
}

matrix_t* transpose(matrix_t* a) {
  // #ifdef MKL
  // matrix_t* ret = matrix_clone(a);
  // mkl_dimatcopy('r', 't', a->rows, a->cols, 1, ret->data, a->cols, a->rows);
  // ret->cols = a->rows;
  // ret->rows = a->cols;
  // return ret;
  // #endif
  matrix_t* new_mat = new_matrix(a->cols, a->rows);
  for (int i = 0; i < new_mat->rows; ++i) {
    for (int j = 0; j < new_mat->cols; ++j) {
      new_mat->data[i*new_mat->cols+j] = a->data[j*a->cols+i];
    }
  }
  return new_mat;
}

float mean(matrix_t* a) {
  assert(a->rows > 0 && a->cols >0);
  float sum = 0;
  // #ifndef MKL
  for (int i = 0; i < a->rows*a->cols; ++i) sum += a->data[i];
  // #else
  // sum = cblas_dasum(a->rows*a->cols, a->data, 1);
  // #endif
  return sum / (float)(a->rows * a->cols);
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
  // printf("augmenting space rows %d cols %d or %d oc %d omax %d\n", rows, cols, t->rows, t->cols, t->max_size);
  assert(rows >= t->rows);
  assert(cols >= t->cols);

  t->max_size = rows * cols;
  #ifndef MKL
  t->data = realloc(t->data, rows*cols*sizeof(float));
  #else
  t->data = mkl_realloc(t->data, rows*cols*sizeof(float));
  #endif
  if (!t->data) {
    printf("[AUGMENT_SPACE] error reallocating memory");
    exit(1);
  }
  return 1;
}

int any_larger(matrix_t* t, float thres) {
  for (int i = 0; i < t->rows*t->cols; ++i) {
    if (t->data[i] > thres) {
      return 1;
    }
  }
  return 0;
}

int copy_matrix(matrix_t* dst, matrix_t* src) {
  if (dst->max_size < src->rows*src->cols) {
    printf("from %d to %d\n",src->rows*src->cols,dst->max_size);
  }
  assert(dst->max_size >= src->rows*src->cols);
  // #ifndef MKL
  memcpy(dst->data, src->data, src->cols*src->rows*sizeof(float));
  // #else
  // MKL_INT n = src->rows*src->cols;
  // MKL_INT incx = 1;
  // MKL_INT incy = 1;
  // cblas_dcopy(n, src->data, incx, dst->data, incy);
  // #endif
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
  float* new_data = calloc(t->rows*t->cols, sizeof(float));
  #else
  float* new_data = MKL_calloc(t->rows*t->cols, sizeof(float), 64);
  #endif
  for (int i = 0; i < t->rows; ++i) memcpy(new_data+i*t->cols, t->data+idx[i]*t->cols, t->cols*sizeof(float));
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
  new_m->data = calloc(rows*cols, sizeof(float));
  #else
  new_m->data = mkl_calloc(rows*cols, sizeof(float), 64);
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
  memcpy(ret->data, t->data+t_start, t_size*sizeof(float));
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
    memcpy(ret->data+(i*ret->cols), t->data+(i*t->cols+start), row_size*sizeof(float));
  }
  return ret;
}

int xavier_init(matrix_t* a, float gain) {
  float low = -gain * sqrt((float)6 / (float)(a->rows+a->cols));
  float high = gain * sqrt((float)6 / (float)(a->rows+a->cols));
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

int ones_init(matrix_t* a) {
  for (int i = 0; i < a->cols*a->rows; ++i) a->data[i] = 1;
  return 1;
}

matrix_t* normalize(matrix_t* t) {
  matrix_t* ret = new_matrix(2, t->cols);
  for (int i = 0; i < t->cols; ++i) {
    float max = t->data[i];
    float min = t->data[i];
    for (int j = 0; j < t->rows; ++j) {
      float curr_num = t->data[j*t->cols+i];
      if (curr_num > max) {
        max = curr_num;
      }
      if (curr_num < min) {
        min = curr_num;
      }
    }
    ret->data[i] = max;
    ret->data[t->cols+i] = min;
    if (max == min) {
      continue;
    }
    for (int j = 0; j < t->rows; ++j) {
      t->data[j*t->cols+i] = (t->data[j*t->cols+i] - min) / (max - min);
    }
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

int matrix_log(matrix_t* a) {
  for (int i = 0; i < a->rows*a->cols; ++i) a->data[i] = log(a->data[i]);
  return 1;
}

matrix_t* matrix_row_argmax(matrix_t* a) {
  matrix_t* ret = new_matrix(a->rows, 1);
  for (int i = 0; i < a->rows; ++i) {
    int max = 0;
    for (int j = 0; j < a->cols; ++j) {
      if (a->data[i*a->cols+j] > a->data[i*a->cols+max]) max = j;
    }
    ret->data[i] = max;
  }
  return ret;
}

matrix_t* matrix_sum(matrix_t* a, int axis) {
  // 1 horizontal, 0 vertical, 2 all
  assert(axis == 0 || axis == 1 || axis == 2);
  matrix_t* ret;
  switch (axis) {
  case 0: {
    ret = new_matrix(1, a->cols);
    initialize(ret, zeros);
    for (int i = 0; i < a->rows*a->cols; ++i) {
      ret->data[i%ret->cols] += a->data[i];
    }
    break;
  }
  case 1: {
    ret = new_matrix(a->rows, 1);
    initialize(ret, zeros);
    for (int i = 0; i < a->rows*a->cols; ++i) {
      ret->data[i/a->cols] += a->data[i];
    }
    break;
  }
  case 2: {
    ret = new_matrix(1,1);
    ret->data[0] = 0;
    for (int i = 0; i < a->rows*a->cols; ++i) {
      ret->data[0] += a->data[i];
    }
    break;
  }
  default:
    printf("[MATRIX_SUM] wrong axis provided\n");
    exit(1);
    break;
  }
  return ret;
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
      memcpy(ret->data+(i*cols), a->data+(i*a->cols), a->cols*sizeof(float));
      memcpy(ret->data+(i*cols)+a->cols, b->data+(i*b->cols), b->cols*sizeof(float));
    }
  } else {
    assert(a->cols == b->cols);
    int rows = a->rows + b->rows;
    int cols = a->cols;
    ret = new_matrix(rows, cols);
    //memcpy(ret->data, a->data, a->cols*a->rows*sizeof(float));
    //memcpy(ret->data+(a->rows*a->cols), b->data, b->cols*b->rows*sizeof(float));
    for (int i = 0; i < a->rows; ++i) memcpy(ret->data+(i*cols), a->data+(i*cols), cols*sizeof(float));
    for (int i = 0; i < b->rows; ++i) memcpy(ret->data+((i+a->rows)*cols), b->data+(i*cols), cols*sizeof(float));
  }
  return ret;
}

matrix_t* matrix_clone(matrix_t* a) {
  assert(a);
  matrix_t* ret = new_matrix(a->rows, a->cols);
  copy_matrix(ret, a);
  return ret;
}

matrix_t* one_hot_encoding(matrix_t* a, int size) {
  assert(a->cols == 1);
  matrix_t* ret = new_matrix(a->rows, size);
  initialize(ret, zeros);
  for (int i = 0; i < a->rows; ++i) {
    ret->data[i*size+(int)a->data[i]] = 1;
  }
  return ret;
}

int clip(matrix_t* a, float low, float high) {
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
