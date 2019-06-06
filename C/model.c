#include "model.h"
#include <stdlib.h>
#include <assert.h>
#include "matrix_op.h"
#include <stdio.h>
#include "layers.h"
#include "macros.h"
#include <math.h>

static double model_forward(model* m, matrix_t* x, matrix_t* y);

model* init_model(int input_dim) {
  model* new_m = malloc(sizeof(model));
  new_m->input_dim = new_m->output_dim = new_m->max_out = input_dim;
  new_m->num_of_layers = 0;
  new_m->hidden_linears = (layer*)malloc(sizeof(layer));
  new_m->hidden_activations = (layer*)malloc(sizeof(layer));
  new_m->loss_layer.type = no_loss;
  new_m->cache_initialized = 0;
  new_m->optimizer.type = no_opt;
  new_m->param_size = 0;
  return new_m;
}

int add_linear_layer(model* m, int number_of_neurons, layer_type activation) {
  assert(number_of_neurons > 0);
  m->num_of_layers++;
  m->max_out = number_of_neurons > m->max_out ? number_of_neurons : m->max_out;
  // new linear layer
  layer linear_wrapper;
  init_linear(&linear_wrapper, m->output_dim, number_of_neurons);
  
  m->param_size += m->output_dim * number_of_neurons;
  m->param_size += m->output_dim;
  
  // realloc m->hidden_linears to hold one more layer
  m->hidden_linears = realloc(m->hidden_linears, sizeof(layer)*m->num_of_layers);
  m->hidden_linears[m->num_of_layers-1] = linear_wrapper;

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

int compile_model(model* m, layer_type loss, optimizer_type opt_type) {
  // cache size is decided at fit time
  // initialize loss layer
  layer loss_wrapper;
  switch (loss) {
    case mse_loss: {
      mse_loss_layer new_mse_loss;
      new_mse_loss.cache_pred = NULL;
      new_mse_loss.cache_target = NULL;
      loss_wrapper.type = mse_loss;
      loss_wrapper.data.m = new_mse_loss;
      break;
    }
    case no_loss:
      loss_wrapper.type = no_loss;
      break;
    default:
      printf("[COMPILE_MODEL] unrecognized loss\n");
      exit(1);
  }
  m->loss_layer = loss_wrapper;

  // initialize optimizer
  optimizer opt_wrapper;
  switch (opt_type) {
    case adam: {
      init_adam(m);
      break;
    }
    case sgd: {
      sgd_optimizer new_sgd;
      new_sgd.learning_rate = SGD_LR;
      opt_wrapper.type = sgd;
      opt_wrapper.cache.s = new_sgd;
      m->optimizer = opt_wrapper;
      break;
    }
    case no_opt: {
      printf("[CAVEAT] Optimizer not specified\n");
      break;
    }
    default:
      break;
  }
  return 1;
}

int print_network(model* m) {
  char* names[] = {"tanh","relu", "linear", "sigmoid", "identity", "mse_loss", "no_loss"};
  char* opt_names[] = {"sgd", "adam", "no_opt"};
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
  printf(" %s %s\n", names[m->loss_layer.type], opt_names[m->optimizer.type]);
  printf("---------------------------------------\n");
  return 1;
}

double fit(model* m, matrix_t* x, matrix_t* y, int batch_size, int epoch, double learning_rate, int shuffle, int auto_update) {

  assert(x->rows == y->rows);
  if (!m->cache_initialized && !init_caches(m, x->rows)) {
    printf("[INIT_CACHES] failed to initialize caches\n");
    exit(1);
  }
  double final_loss = 0;

  for (int epc = 0; epc < epoch; ++epc) {
    #ifdef RUN_TEST
    clock_t e_start = clock(), e_diff;
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
    double loss = 0;
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
      if (auto_update) {
        perform_update(m, learning_rate);
      } else if (batch_size < x->rows || epoch != 1) {
        printf("[Warning] Model is not updating, entering next loop ... batch size: %d, x_size: %d, epoch: %d\n", batch_size, x->rows, epoch);
      }

      start = start + curr_batch;
    }
    if (epc == epoch - 1) {
      final_loss = loss;
    }
    #ifdef RUN_TEST
    e_diff = clock() - e_start;
    int msec = e_diff * 1000 / CLOCKS_PER_SEC;
    printf("%f, time: %d ms", loss, msec);
    fflush(stdout);
    if (loss > 1000) {
      printf("Anomalous loss %f\n", loss);
      exit(1);
    }
    #else
    //printf("%f\n", loss);
    #endif
  }
  #ifdef RUN_TEST
  printf("\n");
  #endif
  return final_loss;
}

int init_caches(model* m, int batch_size) {
  if (m->cache_initialized) {
    return 1;
  }
  m->cache_initialized = 1;
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
  if (!m->cache_initialized) {
    printf("[CAVEAT] Automatically initializing cache with size %d\n", x->rows);
    init_caches(m, x->rows);
  }
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
    free_layer(m->hidden_linears[i]);
  }
  free_layer(m->loss_layer);
  free(m->hidden_linears);
  free(m->hidden_activations);
  free_optimizer(m->optimizer);
  free(m);
  return 1;
}
