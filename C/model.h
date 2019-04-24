#ifndef MODEL_H
#define MODEL_H
#include "layers.h"

typedef struct _model {
  int input_dim;
  int output_dim;
  layer loss_layer;
  layer* hidden_linears;
  layer* hidden_activations;
} model;

model* init_model(int input_dim, int number_of_layers);
void add_linear_layer()

// initialize all the memories for cache according to batch size
// run through all the sample and update model
void fit(model* m, int batch_size, int epoch, double learning_rate);

#endif

