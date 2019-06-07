#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "matrix_op.h"
#include <sys/time.h>

#define BUFFER_SIZE 256

float rand_uniform(float low, float high);
matrix_t* load_data(char* filename);
matrix_t* rand_normal(int size);
matrix_t* trunc_normal(int size, float high, float low);
void timer_reset(struct timeval* t);
float timer_check(struct timeval* t);
#ifdef __cplusplus
}
#endif

#endif

