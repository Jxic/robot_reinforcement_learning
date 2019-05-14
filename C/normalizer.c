#include "normalizer.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>

normalizer* init_normalizer(int state_dim, double clip_range) {
  normalizer* new_n = malloc(sizeof(normalizer));
  new_n->n = 0;
  new_n->clip_value = clip_range;
  new_n->mean = new_matrix(1, state_dim);
  new_n->mean_diff = new_matrix(1, state_dim);
  new_n->var = new_matrix(1, state_dim);
  initialize(new_n->mean, zeros);
  initialize(new_n->mean_diff, zeros);
  initialize(new_n->var, zeros);
  return new_n;
}

int update_normalizer(normalizer* n, matrix_t** observations, int count) {
  assert(count > 0);
  assert(observations[0]->cols > n->mean->cols);
  int state_dim = n->mean->cols;
 
  for (int i = 0; i < count; ++i) {
    matrix_t* nxt_x = slice_col_wise(observations[i], 0, state_dim);

    n->n++;
    matrix_t* last_mean = matrix_clone(n->mean);

    neg(n->mean);
    elem_wise_add(n->mean, nxt_x);
    mult_scalar(n->mean, 1/(double)n->n);
    elem_wise_add(n->mean, last_mean);

    matrix_t* curr_mean = matrix_clone(n->mean);
    neg(last_mean);
    neg(curr_mean);
    elem_wise_add(last_mean, nxt_x);
    elem_wise_add(curr_mean, nxt_x);
    elem_wise_mult(curr_mean, last_mean);
    elem_wise_add(n->mean_diff, curr_mean);

    free_matrix(n->var);
    matrix_t* curr_mean_diff = matrix_clone(n->mean_diff);
    mult_scalar(curr_mean_diff, 1/(double)n->n);
    clip(curr_mean_diff, 1e-2, n->clip_value);
    n->var = curr_mean_diff;

    free_matrix(nxt_x);
    free_matrix(last_mean);
    free_matrix(curr_mean);
  }
  return 1;
}

int normalize_obs(normalizer* n, matrix_t* states) {
  assert(n->mean->cols==states->cols);

  matrix_t* std = matrix_clone(n->var);
  // print_matrix(std, 1);
  square_root(std);
  matrix_t* mean = matrix_clone(n->mean);
  // print_matrix(mean, 1);
  matrix_t* shaped_std = new_matrix(states->rows, states->cols);
  matrix_t* shaped_mean = new_matrix(states->rows, states->cols);
  for (int i = 0; i < states->rows; ++i) {
    memcpy(shaped_std->data+(i*states->cols), std->data, states->cols*sizeof(double));
    memcpy(shaped_mean->data+(i*states->cols), mean->data, states->cols*sizeof(double));
  }
  neg(shaped_mean);
  elem_wise_add(states, shaped_mean);
  inverse(shaped_std);
  elem_wise_mult(states,shaped_std);
  return 1;
}
