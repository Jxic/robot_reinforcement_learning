#ifndef MODEL_H
#define MODEL_H
#include "layers.h"


typedef struct _adam_optimizer {
  int timestamp;
  double beta1;
  double beta2;
  layer* first_moment;
  layer* second_moment;
  double epsilon;
} adam_optimizer;

typedef enum _optimizer {
  sgd, adam
} optimizer;

typedef struct _model {
  int input_dim;
  int output_dim;
  int num_of_layers;
  int max_out;
  layer loss_layer;
  layer* hidden_linears;
  layer* hidden_activations;
  int version;
  int cache_initialzed;
  optimizer opt_type;
  adam_optimizer optimizer;
} model;




model* init_model(int input_dim, optimizer opt_type);
int add_linear_layer(model* m, int number_of_neurons, layer_type activation);
int compile_model(model* m, layer_type loss);
int print_network(model* m);

// initialize all the memories for cache according to batch size
// run through all the sample and update model
double fit(model* m, matrix_t* x, matrix_t* y, int batch_size, int epoch, double learning_rate, int shuffle);
int predict(model* m, matrix_t* x);
double eval(model* m, matrix_t* x, matrix_t* y, matrix_t* min_max);

int model_backward(model* m, matrix_t* grad);
int model_update_sgd(model* m, double learning_rate);
int model_update_adam(model* m, double learning_rate);
int init_caches(model* m, int batch_size);
int free_model(model* m);

#endif

