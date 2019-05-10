#include "optimizer.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "model.h"

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

  // layer adam1;
  //   layer adam2;
  //   init_linear(&adam1, m->output_dim, number_of_neurons);
  //   init_linear(&adam2, m->output_dim, number_of_neurons);
  //   adam1.data.l.cache = new_matrix(1,1);
  //   adam2.data.l.cache = new_matrix(1,1);
  //   m->optimizer.first_moment = realloc(m->optimizer.first_moment, sizeof(layer)*m->num_of_layers);
  //   m->optimizer.second_moment = realloc(m->optimizer.second_moment, sizeof(layer)*m->num_of_layers);
  //   m->optimizer.first_moment[m->num_of_layers-1] = adam1;
  //   m->optimizer.second_moment[m->num_of_layers-1] = adam2;
  //   initialize(adam1.data.l.W, zeros);
  //   initialize(adam1.data.l.b, zeros);
  //   initialize(adam2.data.l.W, zeros);
  //   initialize(adam2.data.l.b, zeros);

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

static int adam_update(model* m) {
  m->optimizer.cache.a.timestamp++;
  adam_optimizer optimizer = m->optimizer.cache.a;
  double learning_rate = optimizer.learning_rate;

  for (int i = 0; i < m->num_of_layers; ++i) {
    layer* nxt_linear = m->hidden_linears + i;
    layer* nxt_fst_moment = optimizer.first_moment + i;
    layer* nxt_snd_moment = optimizer.second_moment + i;

    matrix_t* grad_W = nxt_linear->data.l.grad_W;
    matrix_t* grad_b = nxt_linear->data.l.grad_b;

    // update first moment
    mult_scalar(nxt_fst_moment->data.l.W, optimizer.beta1);
    mult_scalar(nxt_fst_moment->data.l.b, optimizer.beta1);
    
    matrix_t* fst_temp_W = clone(grad_W);
    matrix_t* fst_temp_b = clone(grad_b);
    mult_scalar(fst_temp_W, (1-optimizer.beta1));
    mult_scalar(fst_temp_b, (1-optimizer.beta1));
    elem_wise_add(nxt_fst_moment->data.l.W, fst_temp_W);
    elem_wise_add(nxt_fst_moment->data.l.b, fst_temp_b);

    // update second moment
    mult_scalar(nxt_snd_moment->data.l.W, optimizer.beta2);
    mult_scalar(nxt_snd_moment->data.l.b, optimizer.beta2);

    matrix_t* snd_temp_W = clone(grad_W);
    matrix_t* snd_temp_b = clone(grad_b);
    elem_wise_mult(snd_temp_W, snd_temp_W);
    elem_wise_mult(snd_temp_b, snd_temp_b);
    mult_scalar(snd_temp_W, (1-optimizer.beta2));
    mult_scalar(snd_temp_b, (1-optimizer.beta2));
    elem_wise_add(nxt_snd_moment->data.l.W, snd_temp_W);
    elem_wise_add(nxt_snd_moment->data.l.b, snd_temp_b);

    // compute bias-corrected first moment
    double beta1_exp = pow(optimizer.beta1, optimizer.timestamp);
    double beta2_exp = pow(optimizer.beta2, optimizer.timestamp);
    matrix_t* corrected_fst_W = clone(nxt_fst_moment->data.l.W);
    matrix_t* corrected_fst_b = clone(nxt_fst_moment->data.l.b);
    matrix_t* corrected_snd_W = clone(nxt_snd_moment->data.l.W);
    matrix_t* corrected_snd_b = clone(nxt_snd_moment->data.l.b);


    learning_rate = learning_rate * (sqrt(1-beta2_exp) / (1-beta1_exp));

    square_root(corrected_snd_W);
    square_root(corrected_snd_b);
    
    add_scalar(corrected_snd_W, optimizer.epsilon);
    add_scalar(corrected_snd_b, optimizer.epsilon);
    inverse(corrected_snd_W);
    inverse(corrected_snd_b);
    elem_wise_mult(corrected_fst_W, corrected_snd_W);
    elem_wise_mult(corrected_fst_b, corrected_snd_b);
    mult_scalar(corrected_fst_W, learning_rate);
    mult_scalar(corrected_fst_b, learning_rate);
    
    // if (any_larger(corrected_fst_W, 10)){
    //   printf("gradients\n");
    //   print_matrix(grad_W, 1);
    //   print_matrix(grad_b, 1);
    //   printf("weights\n");
    //   print_matrix(nxt_linear->data.l.W, 1);
    //   print_matrix(nxt_linear->data.l.b, 1);
    //   printf("change\n");
    //   print_matrix(corrected_fst_W,1);
    //   print_matrix(corrected_fst_b,1);
    //   exit(1);
    // }


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
  }
  return 1;
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
