#ifndef MATRIX_OP_H
#define MATRIX_OP_H

#include "data_structures.h"

void dummy();
void initialize(matrix_t* mat, char* initializer);

int elem_wise_add(matrix_t* a, matrix_t* b);
int elem_wise_minus(matrix_t* a, matrix_t* b);
int elem_wise_mult(matrix_t* a, matrix_t* b);
int add_bias(matrix_t* a, matrix_t* b);

matrix_t* matmul(matrix_t* a, matrix_t* b);
matrix_t* transpose(matrix_t* a);

int equal(matrix_t* a, matrix_t* b);


#endif
