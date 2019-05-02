#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "matrix_op.h"

#define BUFFER_SIZE 256

double rand_uniform(double low, double high);
matrix_t* load_data(char* filename);

#ifdef __cplusplus
}
#endif

#endif

