#include "optimizer.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "model.h"
#include <assert.h>
#include <time.h>
#include <pthread.h>

#define NUM_UPDATE_WORKERS 2
pthread_t threads[NUM_UPDATE_WORKERS];
typedef struct update_info_ {
  double** to;
  double* from;
  int start;
  int end;
} update_info;

void* update_weights_with_new(void* args) {
  update_info* update = (update_info*) args;
  int start = update->start;
  int end = update->end;
  double** to = update->to;
  double* from = update->from;

  // printf("setting from %d to %d\n", start, end);

  for (int i = start; i < end; ++i) {
    *to[i] = from[i];
  }
  // printf("worker %d exiting\n", start);
  free(update);
  return 0;
}

void* warm_up(){
  return 0;
}

int init_adam(model* m) {
  // m->optimizer.beta1 = 0.9;
    // m->optimizer.beta2 = 0.999;
    // m->optimizer.epsilon = 0.00000001;
    // m->optimizer.timestamp = 0;
    // m->optimizer.first_moment = (layer*)malloc(sizeof(layer));
    // _m->optimizer.second_moment = (layer*)malloc(sizeof(layer));
  adam_optimizer new_opt;
  new_opt.beta1 = 0.9;
  new_opt.beta2 = 0.999;
  new_opt.epsilon = 0.00000001;
  new_opt.timestamp = 0;
  new_opt.learning_rate = ADAM_LR;
  new_opt.num_of_layers = m->num_of_layers;

  for (int i = 0; i < NUM_UPDATE_WORKERS; ++i) {
    pthread_create(&threads[i], NULL, warm_up, NULL);
  }
  // printf("warmed up\n");

  // int input_dim = m->input_dim;
  // new_opt.first_moment = calloc(m->num_of_layers, sizeof(layer));
  // new_opt.second_moment = calloc(m->num_of_layers, sizeof(layer));
  // for (int i = 0; i < m->num_of_layers; ++i) {
  //   int output_dim = m->hidden_linears[i].data.l.W->cols;
    
  //   layer adam1;
  //   layer adam2;
  //   init_linear(&adam1, input_dim, output_dim);
  //   init_linear(&adam2, input_dim, output_dim);
  //   adam1.data.l.cache = new_matrix(1,1); // placeholder, for convenience of freeing
  //   adam2.data.l.cache = new_matrix(1,1);
  //   new_opt.first_moment[i] = adam1;
  //   new_opt.second_moment[i] = adam2;
  //   initialize(adam1.data.l.W, zeros);
  //   initialize(adam1.data.l.b, zeros);
  //   initialize(adam2.data.l.W, zeros);
  //   initialize(adam2.data.l.b, zeros);
    
  //   input_dim = output_dim;
  // }
  new_opt.first_moment = new_matrix(1, m->param_size);
  new_opt.second_moment = new_matrix(1, m->param_size);

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
  m->optimizer.cache.a.timestamp++;
  adam_optimizer optimizer = m->optimizer.cache.a;
  double learning_rate = optimizer.learning_rate;

  matrix_t* params = new_matrix(1, m->param_size);
  matrix_t* grads = new_matrix(1, m->param_size);

  matrix_t* fst_moment = optimizer.first_moment;
  matrix_t* snd_moment = optimizer.second_moment;

  for (int i = 0; i < m->param_size; ++i) {
    params->data[i] = *optimizer.trainable_params[i];
    grads->data[i] = *optimizer.trainable_params_g[i];
  }

  mult_scalar(fst_moment, optimizer.beta1);
  matrix_t* grad_tmp = matrix_clone(grads);
  mult_scalar(grad_tmp, (1-optimizer.beta1));
  elem_wise_add(fst_moment, grad_tmp);

  free_matrix(grad_tmp);

  mult_scalar(snd_moment, optimizer.beta2);
  grad_tmp = matrix_clone(grads);
  elem_wise_mult(grad_tmp, grad_tmp);
  mult_scalar(grad_tmp, (1-optimizer.beta2));
  elem_wise_add(snd_moment, grad_tmp);

  free_matrix(grad_tmp);

  double beta1_exp = pow(optimizer.beta1, optimizer.timestamp);
  double beta2_exp = pow(optimizer.beta2, optimizer.timestamp);

  matrix_t* corrected_fst = matrix_clone(fst_moment);
  matrix_t* corrected_snd = matrix_clone(snd_moment);

  learning_rate = learning_rate * (sqrt(1-beta2_exp) / (1-beta1_exp));

  square_root(corrected_snd);
  add_scalar(corrected_snd, optimizer.epsilon);
  inverse(corrected_snd);
  elem_wise_mult(corrected_fst, corrected_snd);
  mult_scalar(corrected_fst, learning_rate);

  elem_wise_minus(params, corrected_fst);
  // printf("before joining\n");
  
  for (int j = 0; j < NUM_UPDATE_WORKERS; ++j) {
    pthread_join(threads[j], NULL);
  }
  // printf("after joining\n");

  int i;
  int step_size = m->param_size/NUM_UPDATE_WORKERS;
  int worker_online = 0;
  for (i = 0; i < m->param_size; i += step_size) {
    // *optimizer.trainable_params[i] = params->data[i];
    update_info* new_u = malloc(sizeof(*new_u));
    new_u->to = optimizer.trainable_params;
    new_u->from = params->data;
    new_u->start = i;
    new_u->end = i + step_size > m->param_size ? m->param_size : i + step_size;
    // printf("Starting thread from %d to %d\n", new_u->start, new_u->end);
    pthread_create(&threads[worker_online], NULL, update_weights_with_new, (void*)new_u);
  }

  if (i < m->param_size) {
    for (int j = i; j < m->param_size; ++j) {
      *optimizer.trainable_params[i] = params->data[i];
    }
  }

  // diff = clock() - start;
  // int msec = diff * 1000 / CLOCKS_PER_SEC;
  // printf("Adam took %d ms\n", msec);

  // printf("main exiting\n");

  // for (int i = 0; i < m->num_of_layers; ++i) {
  //   layer* nxt_linear = m->hidden_linears + i;
  //   layer* nxt_fst_moment = optimizer.first_moment + i;
  //   layer* nxt_snd_moment = optimizer.second_moment + i;

  //   matrix_t* grad_W = nxt_linear->data.l.grad_W;
  //   matrix_t* grad_b = nxt_linear->data.l.grad_b;

  //   // update first moment
  //   mult_scalar(nxt_fst_moment->data.l.W, optimizer.beta1);
  //   mult_scalar(nxt_fst_moment->data.l.b, optimizer.beta1);
    
  //   matrix_t* fst_temp_W = matrix_clone(grad_W);
  //   matrix_t* fst_temp_b = matrix_clone(grad_b);
  //   mult_scalar(fst_temp_W, (1-optimizer.beta1));
  //   mult_scalar(fst_temp_b, (1-optimizer.beta1));
  //   elem_wise_add(nxt_fst_moment->data.l.W, fst_temp_W);
  //   elem_wise_add(nxt_fst_moment->data.l.b, fst_temp_b);

  //   // update second moment
  //   mult_scalar(nxt_snd_moment->data.l.W, optimizer.beta2);
  //   mult_scalar(nxt_snd_moment->data.l.b, optimizer.beta2);

  //   matrix_t* snd_temp_W = matrix_clone(grad_W);
  //   matrix_t* snd_temp_b = matrix_clone(grad_b);
  //   elem_wise_mult(snd_temp_W, snd_temp_W);
  //   elem_wise_mult(snd_temp_b, snd_temp_b);
  //   mult_scalar(snd_temp_W, (1-optimizer.beta2));
  //   mult_scalar(snd_temp_b, (1-optimizer.beta2));
  //   elem_wise_add(nxt_snd_moment->data.l.W, snd_temp_W);
  //   elem_wise_add(nxt_snd_moment->data.l.b, snd_temp_b);

  //   // compute bias-corrected first moment
  //   double beta1_exp = pow(optimizer.beta1, optimizer.timestamp);
  //   double beta2_exp = pow(optimizer.beta2, optimizer.timestamp);
  //   matrix_t* corrected_fst_W = matrix_clone(nxt_fst_moment->data.l.W);
  //   matrix_t* corrected_fst_b = matrix_clone(nxt_fst_moment->data.l.b);
  //   matrix_t* corrected_snd_W = matrix_clone(nxt_snd_moment->data.l.W);
  //   matrix_t* corrected_snd_b = matrix_clone(nxt_snd_moment->data.l.b);


  //   learning_rate = learning_rate * (sqrt(1-beta2_exp) / (1-beta1_exp));

  //   square_root(corrected_snd_W);
  //   square_root(corrected_snd_b);
    
  //   add_scalar(corrected_snd_W, optimizer.epsilon);
  //   add_scalar(corrected_snd_b, optimizer.epsilon);
  //   inverse(corrected_snd_W);
  //   inverse(corrected_snd_b);
  //   elem_wise_mult(corrected_fst_W, corrected_snd_W);
  //   elem_wise_mult(corrected_fst_b, corrected_snd_b);
  //   mult_scalar(corrected_fst_W, learning_rate);
  //   mult_scalar(corrected_fst_b, learning_rate);
    
  //   // if (any_larger(corrected_fst_W, 10)){
  //   //   printf("gradients\n");
  //   //   print_matrix(grad_W, 1);
  //   //   print_matrix(grad_b, 1);
  //   //   printf("weights\n");
  //   //   print_matrix(nxt_linear->data.l.W, 1);
  //   //   print_matrix(nxt_linear->data.l.b, 1);
  //   //   printf("change\n");
  //   //   print_matrix(corrected_fst_W,1);
  //   //   print_matrix(corrected_fst_b,1);
  //   //   exit(1);
  //   // }


  //   elem_wise_minus(nxt_linear->data.l.W, corrected_fst_W);
  //   elem_wise_minus(nxt_linear->data.l.b, corrected_fst_b);

  //   // free resources
  //   free_matrix(corrected_fst_W);
  //   free_matrix(corrected_fst_b);
  //   free_matrix(corrected_snd_W);
  //   free_matrix(corrected_snd_b);
  //   free_matrix(snd_temp_b);
  //   free_matrix(snd_temp_W);
  //   free_matrix(fst_temp_b);
  //   free_matrix(fst_temp_W);
  // }
  
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
