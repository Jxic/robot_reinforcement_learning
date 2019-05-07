#ifndef RL_UTILS_H
#define RL_UTILS_H

#include "matrix_op.h"

typedef struct _experience_buffer {
  matrix_t** experiences;
  int max_size;
  int curr_size;
  int end_pos;
} experience_buffer;

experience_buffer* init_experience_buffer(int size);
int store_experience(experience_buffer* exp_buf, matrix_t* new_exp);
int free_experience_buffer(experience_buffer* exp_buf);
matrix_t* sample_experience(experience_buffer* exp_buf, int num);
void print_experiences(experience_buffer* exp_buf);

#endif
