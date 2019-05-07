#include "model.h"
#include <stdlib.h>
#include <assert.h>
#include "matrix_op.h"
#include <stdio.h>
#include "layers.h"
#include "macros.h"
#include <math.h>

static double model_forward(model* m, matrix_t* x, matrix_t* y);

model* init_model(int input_dim, optimizer opt_type) {
  model* new_m = malloc(sizeof(model));
  new_m->input_dim = new_m->output_dim = new_m->max_out = input_dim;
  new_m->num_of_layers = 0;
  new_m->hidden_linears = (layer*)malloc(sizeof(layer));
  new_m->hidden_activations = (layer*)malloc(sizeof(layer));
  new_m->loss_layer.type = no_loss;
  new_m->cache_initialzed = 0;
  new_m->opt_type = opt_type;
  if (opt_type == adam) {
    new_m->optimizer.beta1 = 0.9;
    new_m->optimizer.beta2 = 0.999;
    new_m->optimizer.epsilon = 0.00000001;
    new_m->optimizer.timestamp = 0;
    new_m->optimizer.first_moment = (layer*)malloc(sizeof(layer));
    new_m->optimizer.second_moment = (layer*)malloc(sizeof(layer));
  }
  return new_m;
}

int add_linear_layer(model* m, int number_of_neurons, layer_type activation) {
  assert(number_of_neurons > 0);
  m->num_of_layers++;
  m->max_out = number_of_neurons > m->max_out ? number_of_neurons : m->max_out;
  // new linear layer
  layer linear_wrapper;
  init_linear(&linear_wrapper, m->output_dim, number_of_neurons);
  
  // realloc m->hidden_linears to hold one more layer
  m->hidden_linears = realloc(m->hidden_linears, sizeof(layer)*m->num_of_layers);
  m->hidden_linears[m->num_of_layers-1] = linear_wrapper;

  // add corresponding adam optimizer config
  if (m->opt_type == adam) {
    layer adam1;
    layer adam2;
    init_linear(&adam1, m->output_dim, number_of_neurons);
    init_linear(&adam2, m->output_dim, number_of_neurons);
    adam1.data.l.cache = new_matrix(1,1);
    adam2.data.l.cache = new_matrix(1,1);
    m->optimizer.first_moment = realloc(m->optimizer.first_moment, sizeof(layer)*m->num_of_layers);
    m->optimizer.second_moment = realloc(m->optimizer.second_moment, sizeof(layer)*m->num_of_layers);
    m->optimizer.first_moment[m->num_of_layers-1] = adam1;
    m->optimizer.second_moment[m->num_of_layers-1] = adam2;
    initialize(adam1.data.l.W, zeros);
    initialize(adam1.data.l.b, zeros);
    initialize(adam2.data.l.W, zeros);
    initialize(adam2.data.l.b, zeros);
    
  }

  m->output_dim = number_of_neurons;

  // add activation layer
  // cache size is decided at fit time
  layer activation_wrapper;
  switch (activation)
  {
    case sigmoid: {
      sigmoid_layer new_sigmoid;
      //new_sigmoid.cache = malloc(sizeof(matrix_t));
      new_sigmoid.cache = NULL;
      activation_wrapper.type = sigmoid;
      activation_wrapper.data.s = new_sigmoid;
      break;
    }
    case tanh_: {
      tanh_layer new_tanh;
      //new_tanh.cache = malloc(sizeof(matrix_t));
      new_tanh.cache = NULL;
      activation_wrapper.type = tanh_;
      activation_wrapper.data.t = new_tanh;
      break;
    }
    case relu: {
      relu_layer new_relu;
      //new_relu.cache = malloc(sizeof(matrix_t));
      new_relu.cache = NULL;
      activation_wrapper.type = relu;
      activation_wrapper.data.r = new_relu;
      break;
    }
    case placeholder: {
      placeholder_layer new_placeholder;
      new_placeholder.dummy = NULL;
      activation_wrapper.type = placeholder;
      activation_wrapper.data.p = new_placeholder;
      break;
    }
    default:
      printf("[ADD_LINEAR_LAYER] unrecognized activation\n");
      exit(1);
  }
  // realloc m->hidden_activations to hold new activation
  m->hidden_activations = realloc(m->hidden_activations, sizeof(layer)*m->num_of_layers);
  m->hidden_activations[m->num_of_layers-1] = activation_wrapper;

  return 1;
}

int compile_model(model* m, layer_type loss) {
  // cache size is decided at fit time
  layer loss_wrapper;
  switch (loss) {
    case mse_loss: {
      mse_loss_layer new_mse_loss;
      // new_mse_loss.cache_pred = malloc(sizeof(matrix_t));
      // new_mse_loss.cache_target = malloc(sizeof(matrix_t));
      new_mse_loss.cache_pred = NULL;
      new_mse_loss.cache_target = NULL;
      loss_wrapper.type = mse_loss;
      loss_wrapper.data.m = new_mse_loss;
      break;
    }
    default:
      printf("[COMPILE_MODEL] unrecognized loss\n");
      exit(1);
  }
  m->loss_layer = loss_wrapper;
  return 1;
}

int print_network(model* m) {
  char* names[] = {"tanh","relu", "linear", "sigmoid", "identity", "mse_loss", "no_loss"};
  printf("---------------------------------------\n");
  printf(" input dimension: %d\n", m->input_dim);
  printf(" output dimension: %d\n", m->output_dim);
  printf(" total number of layers: %d\n", m->num_of_layers);
  printf(" max out: %d\n", m->max_out);
  printf("---------------------------------------\n");
  for (int i = 0; i < m->num_of_layers; ++i) {
    printf(" %s(%d -> %d) -> %s \n", names[m->hidden_linears[i].type],
                                  m->hidden_linears[i].data.l.W->rows,
                                  m->hidden_linears[i].data.l.W->cols,
                                  names[m->hidden_activations[i].type]);
  }
  printf(" %s\n", names[m->loss_layer.type]);
  printf("---------------------------------------\n");
  return 1;
}

double fit(model* m, matrix_t* x, matrix_t* y, int batch_size, int epoch, double learning_rate, int shuffle) {

  assert(x->rows == y->rows);
  if (!m->cache_initialzed && !init_caches(m, x->rows)) {
    printf("[INIT_CACHES] failed to initialize caches\n");
    exit(1);
  }
  double final_loss;
  //printf("\n");
  for (int epc = 0; epc < epoch; ++epc) {
    #ifdef RUN_TEST
    printf("\repoch %d: ", epc+1);
    #else
    // printf("epoch %d: ", epc+1);
    #endif
    // shuffle
    if (shuffle) {
      int* idx = shuffle_row_wise(x, 0);
      shuffle_row_wise(y, idx);
      free(idx);
    }
    int data_size = x->rows;
    int start = 0;
    double loss;
    while (start < data_size - 1) {
      int curr_batch = start+batch_size<data_size ? batch_size : data_size-start;
      
      // prepare next batch
      matrix_t* next_batch = slice_row_wise(x, start, start+curr_batch);
      matrix_t* next_target = slice_row_wise(y, start, start+curr_batch);

      augment_space(next_batch, batch_size, m->max_out);

      // one forward and backward pass
      loss = model_forward(m, next_batch, next_target);
      matrix_t* grad = loss_backward(&m->loss_layer);
      model_backward(m, grad);
      free_matrix(grad);
      switch (m->opt_type) {
        case sgd:
          model_update_sgd(m, learning_rate);
          break;
        case adam:
          model_update_adam(m, learning_rate);
          break;
        default:
          break;
      }

      start = start + curr_batch;
    }
    if (epc == epoch - 1) {
      final_loss = loss;
    }
    #ifdef RUN_TEST
    printf("%f", loss);
    fflush(stdout);
    if (loss > 1000) {
      exit(1);
    }
    #else
    //printf("%f\n", loss);
    #endif
  }
  //printf("\n");
  return final_loss;
}

int init_caches(model* m, int batch_size) {
  m->cache_initialzed = 1;
  int last_layer_out = m->input_dim;
  for (int i = 0; i < m->num_of_layers; ++i) {
    m->hidden_linears[i].data.l.cache = new_matrix(batch_size, last_layer_out);
    last_layer_out = m->hidden_linears[i].data.l.W->cols;
    switch (m->hidden_activations[i].type)
    {
      case sigmoid:
        m->hidden_activations[i].data.s.cache = new_matrix(batch_size, last_layer_out);
        break;
      case relu:
        m->hidden_activations[i].data.r.cache = new_matrix(batch_size, last_layer_out);
        break;
      case tanh_:
        m->hidden_activations[i].data.t.cache = new_matrix(batch_size, last_layer_out);   
      default:
        break;
    } 
  }
  switch (m->loss_layer.type) {
    case mse_loss:
      m->loss_layer.data.m.cache_pred = new_matrix(batch_size, last_layer_out);
      m->loss_layer.data.m.cache_target = new_matrix(batch_size, last_layer_out);
      break; 
    default:
      break;
  }
  return 1;
}


int predict(model* m, matrix_t* x) {
  assert(x->rows > 0 && x->cols > 0);
  augment_space(x, x->rows, m->max_out);
  for (int i = 0; i < m->num_of_layers; ++i) {
    if (!forward(m->hidden_linears+i, x)) {
      printf("[MODEL_FORWARD] failed at %dth linear layer\n", i);
      return 0;
    }
    if (!forward(m->hidden_activations+i, x)) {
      printf("[MODEL_FORWARD] failed at %dth activation layer\n", i);
      return 0;
    }
  }
  return 1;
}

static double model_forward(model* m, matrix_t* x, matrix_t* y) {
  if (!predict(m, x)) {
    exit(1);
  }
  // printf("[forward]===========\n");
  // print_matrix(x, 1);
  // print_matrix(y, 1);
  // printf("[forward]===========\n");
  double loss = loss_forward(&m->loss_layer, x, y);
  return loss;
}

int model_backward(model* m, matrix_t* grad) {
  augment_space(grad, grad->rows, m->max_out);
  for (int i = m->num_of_layers-1; i >= 0; --i) {
    if (!backward(m->hidden_activations+i, grad)) {
      printf("[MODEL_BACKWARD] failed at %dth activation layer\n", i);
      exit(1);
    }
    if (!backward(m->hidden_linears+i, grad)) {
      printf("[MODEL_BACKWARD] failed at %dth linear layer\n", i);
      exit(1);
    }
  }
  return 1;
}

int model_update_sgd(model* m, double learning_rate) {
  for(int i = 0; i < m->num_of_layers; i++) {
    if (!update(m->hidden_linears+i, learning_rate)) {
      printf("[MODEL_UPDATE] failed at %dth linear layer\n", i);
    }
  }
  return 1;
}

int model_update_adam(model* m, double learning_rate) {
  m->optimizer.timestamp++;
  adam_optimizer optimizer = m->optimizer;
  //int last_timestamp = m->optimizer.timestamp - 1;
  for (int i = 0; i < m->num_of_layers; ++i) {
    layer* nxt_linear = m->hidden_linears + i;
    layer* nxt_fst_moment = m->optimizer.first_moment + i;
    layer* nxt_snd_moment = m->optimizer.second_moment + i;

    matrix_t* grad_W = nxt_linear->data.l.grad_W;
    matrix_t* grad_b = nxt_linear->data.l.grad_b;
    // print_matrix(grad_W, 1);
    // print_matrix(grad_b, 1);
   
    //printf("updateing first moment %d\n", i);
    // update first moment
    mult_scalar(nxt_fst_moment->data.l.W, optimizer.beta1);
    mult_scalar(nxt_fst_moment->data.l.b, optimizer.beta1);
    
    matrix_t* fst_temp_W = clone(grad_W);
    matrix_t* fst_temp_b = clone(grad_b);
    mult_scalar(fst_temp_W, (1-optimizer.beta1));
    mult_scalar(fst_temp_b, (1-optimizer.beta1));
    elem_wise_add(nxt_fst_moment->data.l.W, fst_temp_W);
    elem_wise_add(nxt_fst_moment->data.l.b, fst_temp_b);
    // print_matrix(nxt_fst_moment->data.l.W, 1);
    // print_matrix(nxt_fst_moment->data.l.W, 1);
    //printf("updating second moment\n");
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
    //printf("updating corrected moment\n");
    // compute bias-corrected first moment
    double beta1_exp = pow(optimizer.beta1, optimizer.timestamp);
    double beta2_exp = pow(optimizer.beta2, optimizer.timestamp);
    matrix_t* corrected_fst_W = clone(nxt_fst_moment->data.l.W);
    matrix_t* corrected_fst_b = clone(nxt_fst_moment->data.l.b);
    matrix_t* corrected_snd_W = clone(nxt_snd_moment->data.l.W);
    matrix_t* corrected_snd_b = clone(nxt_snd_moment->data.l.b);
    // print_matrix(corrected_snd_W, 1);
    // print_matrix(corrected_snd_b, 1);
    // printf("%f %f\n", beta1_exp, beta2_exp);
    
    // mult_scalar(corrected_fst_W, 1/(double)(1-beta1_exp));
    // mult_scalar(corrected_fst_b, 1/(double)(1-beta1_exp));
    // mult_scalar(corrected_snd_W, 1/(double)(1-beta2_exp));
    // mult_scalar(corrected_snd_b, 1/(double)(1-beta2_exp));

    learning_rate = learning_rate * (sqrt(1-beta2_exp) / (1-beta1_exp));
    // printf("%f \n", 3.2e-318 * 1.2e-318 );
    // exit(1);
    //printf("updating params\n");
    // update params
    // print_matrix(corrected_snd_W, 1);
    // print_matrix(corrected_snd_b, 1);
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
    
    if (any_larger(corrected_fst_W, 10)){
      printf("gradients\n");
      print_matrix(grad_W, 1);
      print_matrix(grad_b, 1);
      printf("weights\n");
      print_matrix(nxt_linear->data.l.W, 1);
      print_matrix(nxt_linear->data.l.b, 1);
      printf("change\n");
      print_matrix(corrected_fst_W,1);
      print_matrix(corrected_fst_b,1);
      exit(1);
    }
    //exit(1);
    // printf("weight\n");
    // print_matrix(nxt_linear->data.l.W, 1);
    // print_matrix(nxt_linear->data.l.b, 1);

    elem_wise_minus(nxt_linear->data.l.W, corrected_fst_W);
    elem_wise_minus(nxt_linear->data.l.b, corrected_fst_b);
    // printf("weight + change\n");
    // print_matrix(nxt_linear->data.l.W, 1);
    // print_matrix(nxt_linear->data.l.b, 1);

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

double eval(model* m, matrix_t* x, matrix_t* y, matrix_t* min_max) {
  double sum = 0;
  matrix_t* min_max_y = slice_col_wise(min_max, x->cols, min_max->cols);
  augment_space(x, x->rows, m->max_out);
  predict(m, x);
  scale(x, min_max_y);
  scale(y, min_max_y);
  sum = loss_forward(&m->loss_layer, x, y);
  return sum;
}

int free_model(model* m) {
  for (int i = 0; i < m->num_of_layers; ++i) {
    free_layer(m->hidden_activations[i]);
    free_layer(m->optimizer.first_moment[i]);
    free_layer(m->optimizer.second_moment[i]);
    free_layer(m->hidden_linears[i]);
  }
  free_layer(m->loss_layer);
  free(m->hidden_linears);
  free(m->hidden_activations);
  free(m->optimizer.first_moment);
  free(m->optimizer.second_moment);
  free(m);
  return 1;
}
