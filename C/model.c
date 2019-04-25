#include "model.h"
#include <stdlib.h>
#include <assert.h>
#include "matrix_op.h"
#include <stdio.h>
#include "layers.h"


model* init_model(int input_dim) {
  model* new_m = malloc(sizeof(model));
  new_m->input_dim = new_m->output_dim = input_dim;
  new_m->num_of_layers = 0;
  new_m->hidden_linears = (layer*)malloc(sizeof(layer));
  new_m->hidden_activations = (layer*)malloc(sizeof(layer));
  return new_m;
}

int add_linear_layer(model* m, int number_of_neurons, layer_type activation) {
  assert(number_of_neurons > 0);
  m->num_of_layers++;

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
      printf("[ADD_LINEAR_LAYER] unrecognized activation");
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
      printf("[COMPILE_MODEL] unrecognized loss");
      exit(1);
  }
  m->loss_layer = loss_wrapper;
  return 1;
}

int print_network(model* m) {
  char* names[] = {"relu", "linear", "sigmoid", "identity", "mse_loss"};
  printf("---------------------------------------\n");
  printf("input dimension: %d\n", m->input_dim);
  printf("output dimension: %d\n", m->output_dim);
  printf("total number of layers: %d\n", m->num_of_layers);
  printf("---------------------------------------\n");
  for (int i = 0; i < m->num_of_layers; ++i) {
    printf(" %s(%d) -> %s -> ", names[m->hidden_linears[i].type],
                                m->hidden_linears[i].data.l.W->cols,
                                  names[m->hidden_activations[i].type]);
  }
  printf("%s\n", names[m->loss_layer.type]);
  printf("---------------------------------------\n");
  return 1;
}

