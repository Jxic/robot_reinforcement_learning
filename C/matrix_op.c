#include "matrix_op.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "macros.h"

void dummy(){
  printf("DUMMY FUNCTION");
}

int elem_wise_op_dimension_check(int* dimension_a, int* dimension_b){
  return 1;
}

int matmul_dimension_check(int* dimension_a, int* dimension_b){
  return 1;
}

int equal(matrix_t* a, matrix_t* b){
  if(a->size != b->size) {
    #ifndef RUN_TEST
    printf("[MATRICES NOT EQUAL] size is different");
    #endif
    return 0;
  }
  if(memcmp(a->data, b->data, (a->size)*sizeof(double))){
    #ifndef RUN_TEST
    printf("[MATRICES NOT EQUAL] data is different");
    #endif
    return 0;
  }
  if(memcmp(a->dimensions, b->dimensions, (*a->dimensions)*sizeof(int))){
    #ifndef RUN_TEST
    printf("[MATRICES NOT EQUAL] dimension is different");
    #endif
    return 0;
  }
  if(memcmp(a->dimensions, b->dimensions, (*b->dimensions)*sizeof(int))){
    #ifndef RUN_TEST
    printf("[MATRICES NOT EQUAL] dimension is different");
    #endif
    return 0;
  }
  return 1;
}

matrix_t* elem_wise_add(matrix_t* a, matrix_t* b){

  int* dimension_a = a->dimensions;
  int* dimension_b = b->dimensions;

  if(!elem_wise_op_dimension_check(dimension_a, dimension_b)) {
    printf("[ELEM_WISE_ADD_DIMENSION UNMATCHED] %d, %d\n", *dimension_a, *dimension_b);
    exit(1);
  }

  matrix_t* tuple[2] = {a, b};

  int unbalanced = 0;
  if(*dimension_a > *dimension_b){
    unbalanced = 1;
  } else if(*dimension_b > *dimension_a){
    unbalanced = 2;
  }

  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* new_data;
  int* new_dimensions;
  int new_size;

  if(unbalanced){
    new_data = calloc(tuple[unbalanced-1]->size, sizeof(double));
    new_dimensions = calloc(*(tuple[unbalanced-1]->dimensions) + 1, sizeof(int));
    memcpy(new_dimensions, tuple[unbalanced-1]->dimensions, sizeof(int) * (*(tuple[unbalanced-1]->dimensions) + 1));
    new_size = tuple[unbalanced-1]->size;
  } else {
    new_data = calloc(tuple[0]->size, sizeof(double));
    new_dimensions = calloc(*(tuple[0]->dimensions) + 1, sizeof(int));
    memcpy(new_dimensions, tuple[0]->dimensions, sizeof(int) * (*(tuple[0]->dimensions) + 1));
    new_size = tuple[unbalanced-1]->size;
  }

  new_mat->dimensions = new_dimensions;
  new_mat->data = new_data;
  new_mat->size = new_size;

  if(unbalanced){
    int outer_loop_count = tuple[unbalanced-1]->dimensions[1];
    double* larger_mat = tuple[unbalanced-1]->data;
    int smaller_idx = unbalanced == 1 ? 1 : 0;
    //printf("%d, %d\n", unbalanced, smaller_idx);
    double* smaller_mat = tuple[smaller_idx]->data;
    int inner_loop_count = tuple[smaller_idx]->size;
    for(int i = 0; i < outer_loop_count; ++i){
      for(int j = 0; j < inner_loop_count; ++j){
        new_data[i*inner_loop_count+j] = smaller_mat[j] + larger_mat[i*inner_loop_count+j];
        //printf("%d, %f, %f, %f\n", i*inner_loop_count+j, new_data[i*inner_loop_count+j], smaller_mat[j], larger_mat[i*inner_loop_count+j]);
      }
    }
  } else {
    int loop_count = tuple[0]->size;
    double* mat_a = a->data;
    double* mat_b = b->data;
    for(int i = 0; i < loop_count; ++i){
      new_data[i] = mat_a[i] + mat_b[i];
    }
  }
  return new_mat;
}

matrix_t* elem_wise_minus(matrix_t* a, matrix_t* b){

  int* dimension_a = a->dimensions;
  int* dimension_b = b->dimensions;

  if(!elem_wise_op_dimension_check(dimension_a, dimension_b)) {
    printf("[ELEM_WISE_MINUS_DIMENSION UNMATCHED] %d, %d\n", *dimension_a, *dimension_b);
    exit(1);
  }

  matrix_t* tuple[2] = {a, b};

  int unbalanced = 0;
  if(*dimension_a > *dimension_b){
    unbalanced = 1;
  } else if(*dimension_b > *dimension_a){
    unbalanced = 2;
  }

  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* new_data;
  int* new_dimensions;
  int new_size;

  if(unbalanced){
    new_data = calloc(tuple[unbalanced-1]->size, sizeof(double));
    new_dimensions = calloc(*(tuple[unbalanced-1]->dimensions) + 1, sizeof(int));
    memcpy(new_dimensions, tuple[unbalanced-1]->dimensions, sizeof(int) * (*(tuple[unbalanced-1]->dimensions) + 1));
    new_size = tuple[unbalanced-1]->size;
  } else {
    new_data = calloc(tuple[0]->size, sizeof(double));
    new_dimensions = calloc(*(tuple[0]->dimensions) + 1, sizeof(int));
    memcpy(new_dimensions, tuple[0]->dimensions, sizeof(int) * (*(tuple[0]->dimensions) + 1));
    new_size = tuple[unbalanced-1]->size;
  }

  new_mat->dimensions = new_dimensions;
  new_mat->data = new_data;
  new_mat->size = new_size;

  if(unbalanced){
    int outer_loop_count = tuple[unbalanced-1]->dimensions[1];
    double* larger_mat = tuple[unbalanced-1]->data;
    int smaller_idx = unbalanced == 1 ? 1 : 0;
    //printf("%d, %d\n", unbalanced, smaller_idx);
    double* smaller_mat = tuple[smaller_idx]->data;
    int inner_loop_count = tuple[smaller_idx]->size;
    for(int i = 0; i < outer_loop_count; ++i){
      for(int j = 0; j < inner_loop_count; ++j){
        new_data[i*inner_loop_count+j] = larger_mat[i*inner_loop_count+j] - smaller_mat[j];
        //printf("%d, %f, %f, %f\n", i*inner_loop_count+j, new_data[i*inner_loop_count+j], smaller_mat[j], larger_mat[i*inner_loop_count+j]);
      }
    }
  } else {
    int loop_count = tuple[0]->size;
    double* mat_a = a->data;
    double* mat_b = b->data;
    for(int i = 0; i < loop_count; ++i){
      new_data[i] = mat_a[i] - mat_b[i];
    }
  }
  return new_mat;
}

matrix_t* elem_wise_mult(matrix_t* a, matrix_t* b){

  int* dimension_a = a->dimensions;
  int* dimension_b = b->dimensions;

  if(!elem_wise_op_dimension_check(dimension_a, dimension_b)) {
    printf("[ELEM_WISE_MULT_DIMENSION UNMATCHED] %d, %d\n", *dimension_a, *dimension_b);
    exit(1);
  }

  matrix_t* tuple[2] = {a, b};

  int unbalanced = 0;
  if(*dimension_a > *dimension_b){
    unbalanced = 1;
  } else if(*dimension_b > *dimension_a){
    unbalanced = 2;
  }

  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* new_data;
  int* new_dimensions;
  int new_size;

  if(unbalanced){
    new_data = calloc(tuple[unbalanced-1]->size, sizeof(double));
    new_dimensions = calloc(*(tuple[unbalanced-1]->dimensions) + 1, sizeof(int));
    memcpy(new_dimensions, tuple[unbalanced-1]->dimensions, sizeof(int) * (*(tuple[unbalanced-1]->dimensions) + 1));
    new_size = tuple[unbalanced-1]->size;
  } else {
    new_data = calloc(tuple[0]->size, sizeof(double));
    new_dimensions = calloc(*(tuple[0]->dimensions) + 1, sizeof(int));
    memcpy(new_dimensions, tuple[0]->dimensions, sizeof(int) * (*(tuple[0]->dimensions) + 1));
    new_size = tuple[unbalanced-1]->size;
  }

  new_mat->dimensions = new_dimensions;
  new_mat->data = new_data;
  new_mat->size = new_size;

  if(unbalanced){
    int outer_loop_count = tuple[unbalanced-1]->dimensions[1];
    double* larger_mat = tuple[unbalanced-1]->data;
    int smaller_idx = unbalanced == 1 ? 1 : 0;
    //printf("%d, %d\n", unbalanced, smaller_idx);
    double* smaller_mat = tuple[smaller_idx]->data;
    int inner_loop_count = tuple[smaller_idx]->size;
    for(int i = 0; i < outer_loop_count; ++i){
      for(int j = 0; j < inner_loop_count; ++j){
        new_data[i*inner_loop_count+j] = smaller_mat[j] * larger_mat[i*inner_loop_count+j];
        //printf("%d, %f, %f, %f\n", i*inner_loop_count+j, new_data[i*inner_loop_count+j], smaller_mat[j], larger_mat[i*inner_loop_count+j]);
      }
    }
  } else {
    int loop_count = tuple[0]->size;
    double* mat_a = a->data;
    double* mat_b = b->data;
    for(int i = 0; i < loop_count; ++i){
      new_data[i] = mat_a[i] * mat_b[i];
    }
  }
  return new_mat;
}

matrix_t* elem_wise_div(matrix_t* a, matrix_t* b){

  int* dimension_a = a->dimensions;
  int* dimension_b = b->dimensions;

  if(!elem_wise_op_dimension_check(dimension_a, dimension_b)) {
    printf("[ELEM_WISE_DIV_DIMENSION UNMATCHED] %d, %d\n", *dimension_a, *dimension_b);
    exit(1);
  }

  matrix_t* tuple[2] = {a, b};

  int unbalanced = 0;
  if(*dimension_a > *dimension_b){
    unbalanced = 1;
  } else if(*dimension_b > *dimension_a){
    unbalanced = 2;
  }

  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* new_data;
  int* new_dimensions;
  int new_size;

  if(unbalanced){
    new_data = calloc(tuple[unbalanced-1]->size, sizeof(double));
    new_dimensions = calloc(*(tuple[unbalanced-1]->dimensions) + 1, sizeof(int));
    memcpy(new_dimensions, tuple[unbalanced-1]->dimensions, sizeof(int) * (*(tuple[unbalanced-1]->dimensions) + 1));
    new_size = tuple[unbalanced-1]->size;
  } else {
    new_data = calloc(tuple[0]->size, sizeof(double));
    new_dimensions = calloc(*(tuple[0]->dimensions) + 1, sizeof(int));
    memcpy(new_dimensions, tuple[0]->dimensions, sizeof(int) * (*(tuple[0]->dimensions) + 1));
    new_size = tuple[unbalanced-1]->size;
  }

  new_mat->dimensions = new_dimensions;
  new_mat->data = new_data;
  new_mat->size = new_size;

  if(unbalanced){
    int outer_loop_count = tuple[unbalanced-1]->dimensions[1];
    double* larger_mat = tuple[unbalanced-1]->data;
    int smaller_idx = unbalanced == 1 ? 1 : 0;
    //printf("%d, %d\n", unbalanced, smaller_idx);
    double* smaller_mat = tuple[smaller_idx]->data;
    int inner_loop_count = tuple[smaller_idx]->size;
    for(int i = 0; i < outer_loop_count; ++i){
      for(int j = 0; j < inner_loop_count; ++j){
        new_data[i*inner_loop_count+j] = larger_mat[i*inner_loop_count+j]  / smaller_mat[j];
        //printf("%d, %f, %f, %f\n", i*inner_loop_count+j, new_data[i*inner_loop_count+j], smaller_mat[j], larger_mat[i*inner_loop_count+j]);
      }
    }
  } else {
    int loop_count = tuple[0]->size;
    double* mat_a = a->data;
    double* mat_b = b->data;
    for(int i = 0; i < loop_count; ++i){
      new_data[i] = mat_a[i] / mat_b[i];
    }
  }
  return new_mat;
}

matrix_t* matmul(matrix_t* a, matrix_t* b){
  int* dimension_a = a->dimensions;
  int* dimension_b = b->dimensions;

  if(!matmul_dimension_check(dimension_a, dimension_b)){
    printf("[MATMUL DIMENSION UNMATCHED] %d, %d\n", *dimension_a, *dimension_b);
    exit(1);
  }

  matrix_t* tuple[2] = {a, b};

  int unbalanced = 0;
  if(*dimension_a > *dimension_b){
    unbalanced = 1;
  } else if(*dimension_b > *dimension_a){
    unbalanced = 2;
  }

  matrix_t* new_mat = malloc(sizeof(matrix_t));
  double* new_data;
  int* new_dimensions;
  int new_size;

  if(unbalanced){
    new_dimensions = calloc(*(tuple[unbalanced-1]->dimensions + 1), sizeof(int));
    // if(unbalanced == 1){
    //   new_data = calloc(tuple[0]->dimensions[1] * tuple[1]->dimensions[1] * tuple[0]->dimensions[0], sizeof(double));
    // } else {
    //   new_data = calloc(tuple[0]->dimensions[0] * tuple[1]->dimensions[2] * tuple[1]->dimensions[0], sizeof(double));
    // }
    new_size = tuple[0]->dimensions[unbalanced%2+1] * tuple[1]->dimensions[unbalanced+1] * tuple[unbalanced-1]->dimensions[1];
    new_data = calloc(new_size, sizeof(double));
    new_dimensions[0] = tuple[unbalanced-1]->dimensions[0];
    new_dimensions[1] = tuple[unbalanced-1]->dimensions[1];
    new_dimensions[2] = tuple[0]->dimensions[unbalanced%2+1];
    new_dimensions[3] = tuple[1]->dimensions[unbalanced+1];
  } else {
    new_size = tuple[0]->dimensions[0] * tuple[1]->dimensions[1];
    new_data = calloc(new_size, sizeof(double));
    new_dimensions = calloc(*(tuple[0]->dimensions) + 1, sizeof(int));
    new_dimensions[0] = tuple[0]->dimensions[0];
    new_dimensions[1] = tuple[0]->dimensions[1];
    new_dimensions[2] = tuple[1]->dimensions[2];
  }

  new_mat->dimensions = new_dimensions;
  new_mat->data = new_data;
  new_mat->size = new_size;

  if(unbalanced){
    int outer_loop_count = new_dimensions[1];
    int larger_idx = unbalanced - 1;
    int smaller_idx = unbalanced == 1 ? 1 : 0;
    double* large_mat = tuple[larger_idx]->data;
    int large_mat_cols = tuple[larger_idx]->dimensions[3];
    int large_mat_rows = tuple[larger_idx]->dimensions[2];
    double* small_mat = tuple[smaller_idx]->data;
    int smaller_mat_cols = tuple[smaller_idx]->dimensions[2];
    int smaller_mat_rows = tuple[smaller_idx]->dimensions[1];
    if(unbalanced == 1){
      for(int o = 0; o < outer_loop_count; ++o){
        for(int i = 0; i < large_mat_rows; ++i){
          for(int j = 0; j < large_mat_cols; ++j){
            for(int k = 0; k < large)
          }
        } 
      }
    } else {

    }
  }
}