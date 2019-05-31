#include "normalizer.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <stdio.h>

normalizer* init_normalizer(int state_dim, double clip_range) {
  normalizer* new_n = malloc(sizeof(normalizer));
  new_n->n = 0;
  new_n->clip_value = clip_range;
  new_n->mean = new_matrix(1, state_dim);
  new_n->sum = new_matrix(1, state_dim);
  new_n->std = new_matrix(1, state_dim);
  new_n->sumsq = new_matrix(1, state_dim);
  initialize(new_n->mean, zeros);
  initialize(new_n->sum, zeros);
  initialize(new_n->sumsq, zeros);
  initialize(new_n->std, ones);
  return new_n;
}

int update_normalizer(normalizer* n, matrix_t** observations, int count) {
  // assert(count > 0);
  // assert(observations[0]->cols > n->mean->cols);
  int state_dim = n->mean->cols;
  n->n += count;
  for (int i = 0; i < count; ++i) {
    matrix_t* nxt_x = slice_col_wise(observations[i], 0, state_dim);
    elem_wise_add(n->sum, nxt_x);
    elem_wise_mult(nxt_x, nxt_x);
    elem_wise_add(n->sumsq, nxt_x);
    free_matrix(nxt_x);
  }
  free_matrix(n->mean);
  free_matrix(n->std);
  n->mean = matrix_clone(n->sum);
  mult_scalar(n->mean, 1/(double)n->n);
  n->std = matrix_clone(n->sumsq);
  mult_scalar(n->std, 1/(double)n->n);
  matrix_t* mean = matrix_clone(n->mean);
  elem_wise_mult(mean, mean);
  elem_wise_minus(n->std, mean);
  clip(n->std, 1e-4, 1e4);
  square_root(n->std);
  free_matrix(mean);
  return 1;
}

int normalize_obs(normalizer* n, matrix_t* states) {
  assert(n->mean->cols==states->cols);

  matrix_t* std = matrix_clone(n->std);
  // print_matrix(std, 1);
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
  clip(states, -n->clip_value, n->clip_value);
  free_matrix(shaped_mean);
  free_matrix(shaped_std);
  free_matrix(mean);
  free_matrix(std);
  //print_matrix(states, 0);
  return 1;
}
