#ifndef LAYERS_H
#define LAYERS_H

#include "data_structures.h"

typedef struct _relu_layer {
  matrix_t* cache;
} relu_layer;

typedef struct _linear_layer {
  matrix_t* cache;
} linear_layer;

typedef struct _sigmoid_layer {
  matrix_t* cache;
} sigmoid_layer;

#endif
