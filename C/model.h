#ifndef MODEL_H
#define MODEL_H
#include "layers.h"
#include "optimizer.h"


typedef struct _model {
  int input_dim;
  int output_dim;
  int num_of_layers;
  int max_out;
  layer loss_layer;
  layer* hidden_linears;
  layer* hidden_activations;
  int version;
  int cache_initialized;
  optimizer opt;
  int param_size;
} model;


model* init_model(int input_dim);
int add_linear_layer(model* m, int number_of_neurons, layer_type activation);
int add_conv_layer(model* m, int i_rows, int i_cols, int i_channels, int f_num, int f_size, int stride, int padding, layer_type activation);
int add_max_pool_layer(model* m, int i_rows, int i_cols, int i_channels, int f_size, int stride);
int compile_model(model* m, layer_type loss, optimizer_type opt_type);
int print_network(model* m);

// initialize all the memories for cache according to batch size
// run through all the sample and update model
float fit(model* m, matrix_t* x, matrix_t* y, int batch_size, int epoch, float learning_rate, int shuffle, int auto_update);
int predict(model* m, matrix_t* x);
float eval(model* m, matrix_t* x, matrix_t* y, matrix_t* min_max);

int model_backward(model* m, matrix_t* grad);
int perform_update(model* m, float learning_rate);
int init_adam(model* m);

int init_caches(model* m, int batch_size);
int free_model(model* m);




#endif

