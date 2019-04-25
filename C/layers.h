#ifndef LAYERS_H
#define LAYERS_H

#include "data_structures.h"
#include "matrix_op.h"

typedef struct _relu_layer {
  matrix_t* cache;
} relu_layer;

typedef struct _linear_layer {
  matrix_t* W;
  matrix_t* b;
  matrix_t* grad_W;
  matrix_t* grad_b;
  matrix_t* cache;
} linear_layer;

typedef struct _sigmoid_layer {
  matrix_t* cache;
} sigmoid_layer;

typedef struct _softmax_layer {
  matrix_t* cache;
} softmax_layer;

typedef struct _placeholder_layer {
  matrix_t* dummy;
} placeholder_layer;

typedef struct _mse_loss_layer {
  matrix_t* cache_pred;
  matrix_t* cache_target;
} mse_loss_layer;

typedef union _layer_data {
  relu_layer r;
  linear_layer l;
  sigmoid_layer s;
  placeholder_layer p;
  softmax_layer so;
  mse_loss_layer m;
} layer_data;

typedef enum _layer_type {
  relu, linear, sigmoid, placeholder, mse_loss
} layer_type;

typedef struct _layer {
  layer_type type;
  layer_data data;
} layer;


int forward(layer* l, matrix_t* x);
int backward(layer* l, matrix_t* grad);
int update(layer* l, double learning_rate);

int init_linear(layer* l, int in, int out);

double loss_forward(layer* l, matrix_t* x, matrix_t* target);
matrix_t* loss_backward(layer* l);

#endif
