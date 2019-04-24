#ifndef LAYERS_H
#define LAYERS_H

#include "data_structures.h"

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

typedef union _layer_data {
  relu_layer r;
  linear_layer l;
  sigmoid_layer s;
} layer_data;

typedef enum _layer_type {
  relu, linear, sigmoid
} layer_type;

typedef struct _layer {
  layer_type type;
  layer_data data;
} layer;


int forward(layer* l, matrix_t* x);
int backward(layer* l, matrix_t* grad);
int update(layer* l);

#endif
