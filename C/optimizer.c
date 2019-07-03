#include "macros.h"
#ifdef OPITMIZER_V1
#include "optimizer.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "model.h"
#include <assert.h>
#include <time.h>
#include "utils.h"

int init_adam(model* m) {
  adam_optimizer new_opt;
  new_opt.beta1 = 0.9;
  new_opt.beta2 = 0.999;
  new_opt.epsilon = 0.00000001;
  new_opt.timestamp = 0;
  new_opt.learning_rate = ADAM_LR;
  new_opt.num_of_layers = m->num_of_layers;

  new_opt.first_moment = new_matrix(1, m->param_size);
  new_opt.second_moment = new_matrix(1, m->param_size);
  new_opt.corrected_fst = new_matrix(1, m->param_size);
  new_opt.corrected_snd = new_matrix(1, m->param_size);
  new_opt.grads_container = new_matrix(1, m->param_size);

  initialize(new_opt.first_moment, zeros);
  initialize(new_opt.second_moment, zeros);

  double** trainable_params = calloc(m->param_size, sizeof(*trainable_params));
  double** trainable_params_g = calloc(m->param_size, sizeof(*trainable_params_g));
  new_opt.trainable_params = trainable_params;
  new_opt.trainable_params_g = trainable_params_g;
  int end_pos = 0;
  for (int i = 0; i < m->num_of_layers; ++i) {
    matrix_t* W = m->hidden_linears[i].data.l.W;
    matrix_t* W_g = m->hidden_linears[i].data.l.grad_W;
    matrix_t* b = m->hidden_linears[i].data.l.b;
    matrix_t* b_g = m->hidden_linears[i].data.l.grad_b;
    for (int j = 0; j < W->rows*W->cols; ++j) {
      trainable_params[end_pos] = &W->data[j];
      trainable_params_g[end_pos] = &W_g->data[j];
      end_pos++;
    }
    for (int j = 0; j < b->rows*b->cols; ++j) {
      trainable_params[end_pos] = &b->data[j];
      trainable_params_g[end_pos] = &b_g->data[j];
      end_pos++;
    }
  }
  assert(end_pos == m->param_size);


  m->optimizer.type = adam;
  m->optimizer.cache.a = new_opt;
  return 1;
}

static int sgd_update(model* m);
static int adam_update(model* m);

int perform_update(model* m, double learning_rate) {
  if (learning_rate > 0) {
    switch (m->optimizer.type) {
      case sgd:
        m->optimizer.cache.s.learning_rate = learning_rate;
        break;
      case adam:
        m->optimizer.cache.a.learning_rate = learning_rate;
        break;
      default:
        break;
    }
  }
  switch (m->optimizer.type) {
    case sgd:
      return sgd_update(m);
    case adam:
      return adam_update(m);
    default:
      break;
  }
  return 1;
}

static int sgd_update(model* m) {
  double learning_rate = m->optimizer.cache.s.learning_rate;
  for(int i = 0; i < m->num_of_layers; i++) {
    if (!update(m->hidden_linears+i, learning_rate)) {
      printf("[MODEL_UPDATE] failed at %dth linear layer\n", i);
    }
  }
  return 1;
}

static int adam_update(model* m) {
  // clock_t start = clock(), diff;
  // struct timeval start;
  // timer_reset(&start);
  m->optimizer.cache.a.timestamp++;
  adam_optimizer optimizer = m->optimizer.cache.a;
  double learning_rate = optimizer.learning_rate;

  matrix_t* grads1 = optimizer.grads_container;


  matrix_t* fst_moment = optimizer.first_moment;
  matrix_t* snd_moment = optimizer.second_moment;
  
  // retrieve params
  for (int i = 0; i < m->param_size; ++i) {
    grads1->data[i] = *optimizer.trainable_params_g[i];
  }
  // double rt = timer_check(&start);

  // updating cache
  elem_wise_minus(fst_moment, grads1);
  mult_scalar(fst_moment, optimizer.beta1);
  elem_wise_add(fst_moment, grads1);

  square(grads1);
  elem_wise_minus(snd_moment, grads1);
  mult_scalar(snd_moment, optimizer.beta2);
  elem_wise_add(snd_moment, grads1);


  // calculate update
  double beta1_exp = pow(optimizer.beta1, optimizer.timestamp);
  double beta2_exp = pow(optimizer.beta2, optimizer.timestamp);
  learning_rate = learning_rate * (sqrt(1-beta2_exp) / (1-beta1_exp));

  // double uct = timer_check(&start);

  // matrix_t* corrected_fst = new_matrix(fst_moment->rows, fst_moment->cols);
  // matrix_t* corrected_snd = new_matrix(snd_moment->rows, snd_moment->cols);
  matrix_t* corrected_fst = optimizer.corrected_fst;
  matrix_t* corrected_snd = optimizer.corrected_snd;
  // double mct = timer_check(&start);
  
  square_root(snd_moment, corrected_snd);
  add_scalar(corrected_snd, optimizer.epsilon);
  elem_wise_div(fst_moment, corrected_snd, corrected_fst);
  mult_scalar(corrected_fst, learning_rate);
  // double calc_ut = timer_check(&start);

  // write back
  for (int j = 0; j < m->param_size; ++j) {
    *optimizer.trainable_params[j] -= corrected_fst->data[j];
  }
  // double wb = timer_check(&start);
  // printf("reatrieve %.1f update cache %.1f mct %.1f calculate update %.1f write back %.1f\n", rt, uct, mct, calc_ut, wb);
  // free_matrix(params);

  return 1;
}

int free_optimizer(optimizer o) {
  switch (o.type) {
    case adam: {
      // for (int i = 0; i < o.cache.a.num_of_layers; ++i) {
      //   free_layer(o.cache.a.first_moment[i]);
      //   free_layer(o.cache.a.second_moment[i]);
      // }
      free_matrix(o.cache.a.first_moment);
      free_matrix(o.cache.a.second_moment);
      break;
    }
    default:
      break;
  }
  return 1;
}

#else

#include "optimizer.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "model.h"
#include <pthread.h>
#include <assert.h>

int init_adam(model* m) {

  adam_optimizer new_opt;
  new_opt.beta1 = 0.9;
  new_opt.beta2 = 0.999;
  new_opt.epsilon = 0.00000001;
  new_opt.timestamp = 0;
  new_opt.learning_rate = ADAM_LR;
  new_opt.num_of_layers = m->num_of_layers;

  int input_dim = m->input_dim;
  new_opt.first_moment = calloc(m->num_of_layers, sizeof(layer));
  new_opt.second_moment = calloc(m->num_of_layers, sizeof(layer));
  for (int i = 0; i < m->num_of_layers; ++i) {
    int output_dim = m->hidden_linears[i].data.l.W->cols;
    
    layer adam1;
    layer adam2;
    init_linear(&adam1, input_dim, output_dim);
    init_linear(&adam2, input_dim, output_dim);
    adam1.data.l.cache = new_matrix(1,1); // placeholder, for convenience of freeing
    adam2.data.l.cache = new_matrix(1,1);
    new_opt.first_moment[i] = adam1;
    new_opt.second_moment[i] = adam2;
    initialize(adam1.data.l.W, zeros);
    initialize(adam1.data.l.b, zeros);
    initialize(adam2.data.l.W, zeros);
    initialize(adam2.data.l.b, zeros);
    
    input_dim = output_dim;
  }

  m->optimizer.type = adam;
  m->optimizer.cache.a = new_opt;
  return 1;
}

static int sgd_update(model* m);
static int adam_update(model* m);

int perform_update(model* m, double learning_rate) {
  if (learning_rate > 0) {
    switch (m->optimizer.type) {
      case sgd:
        m->optimizer.cache.s.learning_rate = learning_rate;
        break;
      case adam:
        m->optimizer.cache.a.learning_rate = learning_rate;
        break;
      default:
        break;
    }
  }
  switch (m->optimizer.type) {
    case sgd:
      return sgd_update(m);
    case adam:
      return adam_update(m);
    default:
      break;
  }
  return 1;
}

static int sgd_update(model* m) {
  double learning_rate = m->optimizer.cache.s.learning_rate;
  for(int i = 0; i < m->num_of_layers; i++) {
    if (!update(m->hidden_linears+i, learning_rate)) {
      printf("[MODEL_UPDATE] failed at %dth linear layer\n", i);
    }
  }
  return 1;
}

#define NUM_UPDATE_WORKERS 4
pthread_t threads[NUM_UPDATE_WORKERS];

typedef struct update_info_ {
  model* m;
  int from;
  int to;
} update_info;

void* update_layer(void* arg);

static int adam_update(model* m) {
  // assert(NUM_UPDATE_WORKERS >= m->num_of_layers);
  m->optimizer.cache.a.timestamp++;
  int num_layers = m->num_of_layers;
  int step_size = num_layers/NUM_UPDATE_WORKERS;
  int activate_n = NUM_UPDATE_WORKERS;
  if (!step_size) {
    step_size = 1;
    activate_n = num_layers;
  }
  
  for (int i = 0; i < activate_n; ++i) {
    // *optimizer.trainable_params[i] = params->data[i];
    update_info* new_u = malloc(sizeof(*new_u));
    new_u->m = m;
    new_u->to = i*step_size + step_size;
    new_u->from = i*step_size;
    // printf("Starting thread from %d to %d, total %d\n", new_u->from, new_u->to, activate_n);
    pthread_create(&threads[i], NULL, update_layer, (void*)new_u);
  }
  if (NUM_UPDATE_WORKERS*step_size < num_layers) {
    // printf("doing extra work\n");
    update_info* new_u = malloc(sizeof(*new_u));
    new_u->m = m;
    new_u->to = NUM_UPDATE_WORKERS*step_size + step_size;
    new_u->from = NUM_UPDATE_WORKERS*step_size;
    pthread_t extra_thread;
    pthread_create(&extra_thread, NULL, update_layer, (void*)new_u);
    pthread_join(extra_thread, NULL);
  }
  for (int j = 0; j < activate_n; ++j) {
    pthread_join(threads[j], NULL);
  }
  return 1;
}


void* update_layer(void* arg) {
  update_info* u = (update_info*) arg;
  model* m = u->m;
  int from = u->from;
  int to = u->to;

  adam_optimizer optimizer = m->optimizer.cache.a;
  double learning_rate = optimizer.learning_rate;

  for (int i = from; i < to; ++i) {
    // printf("doing %d\n", i);
    layer* nxt_linear = m->hidden_linears + i;
    layer* nxt_fst_moment = optimizer.first_moment + i;
    layer* nxt_snd_moment = optimizer.second_moment + i;

    matrix_t* grad_W = nxt_linear->data.l.grad_W;
    matrix_t* grad_b = nxt_linear->data.l.grad_b;

    // update first moment
    mult_scalar(nxt_fst_moment->data.l.W, optimizer.beta1);
    mult_scalar(nxt_fst_moment->data.l.b, optimizer.beta1);
    
    matrix_t* fst_temp_W = matrix_clone(grad_W);
    matrix_t* fst_temp_b = matrix_clone(grad_b);
    mult_scalar(fst_temp_W, (1-optimizer.beta1));
    mult_scalar(fst_temp_b, (1-optimizer.beta1));
    elem_wise_add(nxt_fst_moment->data.l.W, fst_temp_W);
    elem_wise_add(nxt_fst_moment->data.l.b, fst_temp_b);

    // update second moment
    mult_scalar(nxt_snd_moment->data.l.W, optimizer.beta2);
    mult_scalar(nxt_snd_moment->data.l.b, optimizer.beta2);

    matrix_t* snd_temp_W = matrix_clone(grad_W);
    matrix_t* snd_temp_b = matrix_clone(grad_b);
    elem_wise_mult(snd_temp_W, snd_temp_W);
    elem_wise_mult(snd_temp_b, snd_temp_b);
    mult_scalar(snd_temp_W, (1-optimizer.beta2));
    mult_scalar(snd_temp_b, (1-optimizer.beta2));
    elem_wise_add(nxt_snd_moment->data.l.W, snd_temp_W);
    elem_wise_add(nxt_snd_moment->data.l.b, snd_temp_b);

    // compute bias-corrected first moment
    
    double beta1_exp = pow(optimizer.beta1, optimizer.timestamp);
    double beta2_exp = pow(optimizer.beta2, optimizer.timestamp);
    matrix_t* corrected_fst_W = matrix_clone(nxt_fst_moment->data.l.W);
    matrix_t* corrected_fst_b = matrix_clone(nxt_fst_moment->data.l.b);
    matrix_t* corrected_snd_W = matrix_clone(nxt_snd_moment->data.l.W);
    matrix_t* corrected_snd_b = matrix_clone(nxt_snd_moment->data.l.b);


    learning_rate = learning_rate * (sqrt(1-beta2_exp) / (1-beta1_exp));

    square_root(corrected_snd_W, corrected_snd_W);
    square_root(corrected_snd_b, corrected_snd_b);
    
    add_scalar(corrected_snd_W, optimizer.epsilon);
    add_scalar(corrected_snd_b, optimizer.epsilon);
    inverse(corrected_snd_W);
    inverse(corrected_snd_b);
    elem_wise_mult(corrected_fst_W, corrected_snd_W);
    elem_wise_mult(corrected_fst_b, corrected_snd_b);
    mult_scalar(corrected_fst_W, learning_rate);
    mult_scalar(corrected_fst_b, learning_rate);


    elem_wise_minus(nxt_linear->data.l.W, corrected_fst_W);
    elem_wise_minus(nxt_linear->data.l.b, corrected_fst_b);

    // free resources
    free_matrix(corrected_fst_W);
    free_matrix(corrected_fst_b);
    free_matrix(corrected_snd_W);
    free_matrix(corrected_snd_b);
    free_matrix(snd_temp_b);
    free_matrix(snd_temp_W);
    free_matrix(fst_temp_b);
    free_matrix(fst_temp_W);
    // printf("finish %d\n", i);
  }
  return 0;
}


int free_optimizer(optimizer o) {
  switch (o.type) {
    case adam: {
      for (int i = 0; i < o.cache.a.num_of_layers; ++i) {
        free_layer(o.cache.a.first_moment[i]);
        free_layer(o.cache.a.second_moment[i]);
      }
      free(o.cache.a.first_moment);
      free(o.cache.a.second_moment);
      break;
    }
    default:
      break;
  }
  return 1;
}
#endif
