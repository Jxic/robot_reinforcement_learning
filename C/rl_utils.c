#include "rl_utils.h"
#include <stdlib.h>
#include <assert.h>
#include "utils.h"
#include <string.h>

experience_buffer* init_experience_buffer(int size) {
  assert(size > 0);
  experience_buffer* new_buffer = (experience_buffer*)malloc(sizeof(experience_buffer));
  new_buffer->experiences = calloc(1, sizeof(matrix_t*));
  new_buffer->max_size = size;
  new_buffer->curr_size = 0;
  return new_buffer;
}

int store_experience(experience_buffer* exp_buf, matrix_t* new_exp) {
  exp_buf->experiences = realloc(exp_buf->experiences, (exp_buf->curr_size+1)*sizeof(matrix_t*));
  exp_buf->experiences[exp_buf->curr_size] = new_exp;
  if (exp_buf->curr_size >= exp_buf->max_size) {
    free_matrix(exp_buf->experiences[0]);
    exp_buf->experiences++;
  }
  return 1;
}

int free_experience_buffer(experience_buffer* exp_buf) {
  for (int i = 0; i < exp_buf->curr_size; ++i) {
    free_matrix(exp_buf->experiences[i]);
  }
  free(exp_buf->experiences);
  free(exp_buf);
  return 1;
}

matrix_t* sample_experience(experience_buffer* exp_buf, int num) {
  assert(num <= exp_buf->max_size);
  matrix_t* ret = new_matrix(num, exp_buf->experiences[0]->cols);
  
  int idx;
  int chosen[exp_buf->curr_size];
  memset(chosen, 0, exp_buf->curr_size*sizeof(int));
  for (int i = 0; i < num; ++i) {
    idx = (int) rand_uniform(-1, exp_buf->curr_size);
    if (idx == -1) idx++;
    if (idx == exp_buf->curr_size) idx--;
    for (int j = idx; j < exp_buf->curr_size; ++j) {
      if (!chosen[j]) {
        idx = j;
        chosen[j] = 1;
        break;
      }
    }
    assert(exp_buf->experiences[idx]->cols == ret->cols);
    memcpy(ret->data+(i*ret->cols), exp_buf->experiences[idx]->data, ret->cols*sizeof(double));
  }
  return ret;
}
