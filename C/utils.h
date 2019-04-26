#ifndef UTILS_H
#define UTILS_H

#include "matrix_op.h"

#define BUFFER_SIZE 256

double rand_uniform(double low, double high);
matrix_t* load_data(char* filename);

#endif

