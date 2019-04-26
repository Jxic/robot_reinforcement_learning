#include "model.h"
#include <stdlib.h>
#include <assert.h>
#include "matrix_op.h"
#include <stdio.h>
#include "layers.h"

static double model_forward(model* m, matrix_t* x, matrix_t* y);
static matrix_t* model_backard(model* m);
static int model_update(model* m, double learning_rate);
static int init_caches(model* m, int batch_size);

model* init_model(int input_dim) {
  model* new_m = malloc(sizeof(model));
  new_m->input_dim = new_m->output_dim = new_m->max_out = input_dim;
  new_m->num_of_layers = 0;
  new_m->hidden_linears = (layer*)malloc(sizeof(layer));
  new_m->hidden_activations = (layer*)malloc(sizeof(layer));
  return new_m;
}

int add_linear_layer(model* m, int number_of_neurons, layer_type activation) {
  assert(number_of_neurons > 0);
  m->num_of_layers++;
  m->max_out = number_of_neurons > m->max_out ? number_of_neurons : m->max_out;
  // new linear layer
  layer linear_wrapper;
  init_linear(&linear_wrapper, m->output_dim, number_of_neurons);
  m->output_dim = number_of_neurons;

  // realloc m->hidden_linears to hold one more layer
  m->hidden_linears = realloc(m->hidden_linears, sizeof(layer)*m->num_of_layers);
  m->hidden_linears[m->num_of_layers-1] = linear_wrapper;

  // add activation layer
  // cache size is decided at fit time
  layer activation_wrapper;
  switch (activation)
  {
    case sigmoid: {
      sigmoid_layer new_sigmoid;
      new_sigmoid.cache = malloc(sizeof(matrix_t));
      activation_wrapper.type = sigmoid;
      activation_wrapper.data.s = new_sigmoid;
      break;
    }
    case relu: {
      relu_layer new_relu;
      new_relu.cache = malloc(sizeof(matrix_t));
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
      new_mse_loss.cache_pred = malloc(sizeof(matrix_t));
      new_mse_loss.cache_target = malloc(sizeof(matrix_t));
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
  char* names[] = {"relu", "linear", "sigmoid", "identity", "mse_loss"};
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

void fit(model* m, matrix_t* x, matrix_t* y, int batch_size, int epoch, double learning_rate, int shuffle) {
  assert(x->rows == y->rows);
  if (!init_caches(m, batch_size)) {
    printf("[INIT_CACHES] failed to initialize caches\n");
    exit(1);
  }
  normalize(x);
  normalize(y);
  for (int epc = 0; epc < epoch; ++epc) {
    printf("epoch %d: ", epc);
    // todo: shuffle dataset and free the original one
    // if (shuffle) {
    //   x = shuffle_matrix_row_wise(x);
    //   y = shuffle_matrix_row_wise(y);
    // }
    int data_size = x->rows;
    int start = 0;
    double loss;
    while (start < data_size - 1) {
      int curr_batch = start+batch_size<data_size ? batch_size : data_size-start;
      
      // prepare next batch
      matrix_t* next_batch = slice_row_wise(x, start, start+curr_batch);
      matrix_t* next_target = slice_row_wise(y, start, start+curr_batch);
      if (contains_nan(next_batch) || contains_nan(next_target)) {
        printf("batch contains nan, %d", start);
      }
      augment_space(next_batch, batch_size, m->max_out);
      // print_matrix(next_batch, 1);
      // print_matrix(next_target, 1);
      // one forward and backward pass
      loss = model_forward(m, next_batch, next_target);
      //printf("%f\n", loss);
      if (loss != loss) {
        printf("epoch %d, start %d, x %d, y %d\n", epc, start, contains_nan(next_batch), contains_nan(next_target));
        exit(1);
      }
      model_backard(m);
      model_update(m, learning_rate);

      start = start + curr_batch;
      //break;
    }
    //break;
    printf("%f\n", loss);
  }
}

static int init_caches(model* m, int batch_size) {
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

static matrix_t* model_backard(model* m) {
  matrix_t* grad = loss_backward(&m->loss_layer);
  //printf("done grad\n");
  augment_space(grad, grad->rows, m->max_out);
  for (int i = m->num_of_layers-1; i >= 0; --i) {
    if (!backward(m->hidden_activations+i, grad)) {
      printf("[MODEL_BACKWARD] failed at %dth activation layer\n", i);
      exit(1);
    }
    //printf("done activation\n");
    if (!backward(m->hidden_linears+i, grad)) {
      printf("[MODEL_BACKWARD] failed at %dth linear layer\n", i);
      exit(1);
    }
    //printf("done linear\n");
  }
  return grad;
}

static int model_update(model* m, double learning_rate) {
  for(int i = 0; i < m->num_of_layers; i++) {
    if (!update(m->hidden_linears+i, learning_rate)) {
      printf("[MODEL_UPDATE] failed at %dth linear layer\n", i);
    }
  }
  return 1;
}

double eval(model* m, matrix_t* x, matrix_t* y) {
  printf("evaluating trained model\n");
  double sum = 0;
  for (int i = 0; i < x->rows; ++i) {
    matrix_t* nxt_x = slice_row_wise(x, i, i+1);
    augment_space(nxt_x, 1, m->max_out);
    matrix_t* nxt_y = slice_row_wise(y, i, i+1);
    double loss = model_forward(m, nxt_x, nxt_y);
    //printf("%f \n", loss);
    sum += loss;
  }
  sum /= (double) x->rows;
  return sum;
}
