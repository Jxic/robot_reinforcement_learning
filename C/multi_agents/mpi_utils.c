#include "mpi_utils.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define DEBUG_MPI 0

static matrix_t* flatten_grad(model* m);
static int set_grad(model* m, matrix_t* grad);
static matrix_t* flatten_weights(model* m);
static int set_weights(model* m, matrix_t* weights);
static matrix_t* flatten(matrix_t** ms, int count);
static matrix_t** rebuild(model* m, matrix_t* flattened);
static int world_size;
static int rank;

int mpi_init() {
  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("Worker %d initiating, total %d\n", rank, world_size);

  return rank;
}

int mpi_perform_update(model* m, double lr, int scale) {
  if (DEBUG_MPI) {
    printf("%d Entering mpi perform update\n", rank);
  }
  if (m->optimizer.cache.a.timestamp % 100 == 0) {
    mpi_check_sync(m);
  }
  matrix_t* l_grads = flatten_grad(m);
  matrix_t* g_grads = new_matrix(l_grads->rows, l_grads->cols);
  MPI_Allreduce(l_grads->data, g_grads->data, l_grads->rows*l_grads->cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (scale) {
    mult_scalar(g_grads, 1/(double)world_size);
  }
  set_grad(m, g_grads);
  perform_update(m, lr);
  free_matrix(l_grads);
  free_matrix(g_grads);
  if (DEBUG_MPI) {
    printf("%d Leaving mpi perform update\n", rank);
  }
  return 1;
}

int mpi_update_normalizer(normalizer* norm, matrix_t** data, int count) {
  if (DEBUG_MPI) {
    printf("%d Entering mpi update normalizer\n", rank);
  }
  update_normalizer(norm, data, count);
  // printf("%d Finished first round of update normalizer\n", rank);
  int l_n = norm->n;
  matrix_t* l_sum = norm->sum;
  matrix_t* l_sumsq = norm->sumsq;

  int g_n;
  matrix_t* g_avg_sum = new_matrix(l_sum->rows, l_sum->cols);
  matrix_t* g_avg_sumsq = new_matrix(l_sumsq->rows, l_sumsq->cols);

  MPI_Allreduce(&l_n, &g_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(l_sum->data, g_avg_sum->data, l_sum->rows*l_sum->cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(l_sumsq->data, g_avg_sumsq->data, l_sumsq->rows*l_sumsq->cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  g_n /= world_size;
  mult_scalar(g_avg_sum, 1/(double)world_size);
  mult_scalar(g_avg_sumsq, 1/(double)world_size);

  free_matrix(norm->sum);
  free_matrix(norm->sumsq);
  
  norm->n = g_n;
  norm->sum = g_avg_sum;
  norm->sumsq = g_avg_sumsq;
  update_normalizer(norm, NULL, 0);

  // norm->n = 0;
  // initialize(norm->sum, zeros);
  // initialize(norm->sumsq, zeros);
  if (DEBUG_MPI) {
    printf("%d Leaving mpi update normalizer\n", rank);
  }
  return 1;
}

int mpi_sync(model* m) {
  if (DEBUG_MPI) {
    printf("%d Entering mpi sync\n", rank);
  }
  matrix_t* w = flatten_weights(m);
  MPI_Bcast(w->data, w->cols*w->rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  set_weights(m, w);
  free_matrix(w);
  if (DEBUG_MPI) {
    printf("%d Leaving mpi sync\n", rank);
  }
  return 1;
}

int mpi_check_sync(model* m) {
  if (DEBUG_MPI) {
    printf("%d Entering mpi check sync\n", rank);
  }
  if (!rank) {
    matrix_t* w = flatten_weights(m);
    MPI_Bcast(w->data, w->cols*w->rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free_matrix(w);
  } else {
    matrix_t* l_w = flatten_weights(m);
    matrix_t* root_w = new_matrix(l_w->rows, l_w->cols);
    MPI_Bcast(root_w->data, l_w->rows*l_w->cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    assert(equal(l_w, root_w));
    free_matrix(root_w);
    free_matrix(l_w);
  }
  if (DEBUG_MPI) {
    printf("%d Leaving mpi check sync\n", rank);
  }
  return 1;
}

int mpi_finalize(int rank) {
  if (DEBUG_MPI) {
    printf("Worker %d exiting ... ", rank);
  }
  MPI_Finalize();
  return 1;
}

// static matrix_t* flatten_grad(model* m);
// static int set_grad(model* m, matrix_t* grad);
// static matrix_t* flatten_weights(model* m);
// static int set_weights(model* m, matrix_t* weights);

static matrix_t* flatten_grad(model* m) {
  matrix_t* ret;

  matrix_t** gs = calloc(2*m->num_of_layers, sizeof(*gs));
  for (int i = 0; i < 2*m->num_of_layers; i+=2) {
    gs[i] = m->hidden_linears[i/2].data.l.grad_W;
    gs[i+1] = m->hidden_linears[i/2].data.l.grad_b;
  }
  
  ret = flatten(gs, 2*m->num_of_layers);
  free(gs);
  return ret;
}

static matrix_t* flatten_weights(model* m) {
  matrix_t* ret;

  matrix_t** ws = calloc(2*m->num_of_layers, sizeof(*ws));
  for (int i = 0; i < 2*m->num_of_layers; i+=2) {
    ws[i] = m->hidden_linears[i/2].data.l.W;
    ws[i+1] = m->hidden_linears[i/2].data.l.b;
  }
  
  ret = flatten(ws, 2*m->num_of_layers);
  free(ws);
  return ret;
}

static int set_grad(model* m, matrix_t* grad) {
  matrix_t** gs = rebuild(m, grad);

  for (int i = 0; i < 2*m->num_of_layers; i+=2) {
    copy_matrix(m->hidden_linears[i/2].data.l.grad_W, gs[i]);
    copy_matrix(m->hidden_linears[i/2].data.l.grad_b, gs[i+1]);
    free_matrix(gs[i]);
    free_matrix(gs[i+1]);
  }

  free(gs);

  return 1;
}

static int set_weights(model* m, matrix_t* weights) {
  matrix_t** ws = rebuild(m, weights);

  for (int i = 0; i < 2*m->num_of_layers; i+=2) {
    copy_matrix(m->hidden_linears[i/2].data.l.W, ws[i]);
    copy_matrix(m->hidden_linears[i/2].data.l.b, ws[i+1]);
    free_matrix(ws[i]);
    free_matrix(ws[i+1]);
  }

  free(ws);

  return 1;
}

// static matrix_t* flatten(matrix_t** ms, int count);
// static matrix_t** rebuild(model* m, matrix_t* flattened);

static matrix_t* flatten(matrix_t** ms, int count) {
  assert(count > 0);
  int size = 0;
  for (int i = 0; i < count; ++i) {
    size += ms[i]->cols*ms[i]->rows;
  }

  int start = 0;
  matrix_t* ret = new_matrix(1, size);
  for (int i = 0; i < count; ++i) {
    memcpy(ret->data+start, ms[i]->data, ms[i]->rows*ms[i]->cols*sizeof(double));
    start += ms[i]->rows*ms[i]->cols;
  }

  return ret;
}

static matrix_t** rebuild(model* m, matrix_t* flattened) {
  assert(flattened->rows == 1);
  assert(flattened->cols > 0);

  matrix_t** ret = calloc(2*m->num_of_layers, sizeof(*ret));
  int start = 0;
  for (int i = 0; i < 2*m->num_of_layers; i+=2) {
    int w_rows = m->hidden_linears[i/2].data.l.W->rows;
    int w_cols = m->hidden_linears[i/2].data.l.W->cols;
    int b_rows = m->hidden_linears[i/2].data.l.b->rows;
    int b_cols = m->hidden_linears[i/2].data.l.b->cols;
    matrix_t* W = new_matrix(w_rows, w_cols);
    matrix_t* b = new_matrix(b_rows, b_cols);
    assert(start < flattened->cols);
    memcpy(W->data, flattened->data+start, w_rows*w_cols*sizeof(double));
    start += w_rows*w_cols;
    assert(start < flattened->cols);
    memcpy(b->data, flattened->data+start, b_rows*b_cols*sizeof(double));
    start += b_rows*b_cols;
    ret[i] = W;
    ret[i+1] = b;
  }
  
  return ret;
}
