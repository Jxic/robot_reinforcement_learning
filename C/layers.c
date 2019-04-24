#include "layers.h"
#include <stdlib.h>
#include <stdio.h>
#include "matrix_op.h"
#include <string.h>
#include <math.h>

// hidden layer neurons forward
static int linear_forward(layer* l, matrix_t* x);
// activation layer forward
static int relu_forward(layer* l, matrix_t* x);
static int sigmoid_forward(layer* l, matrix_t* x);
static int softmax_forward(layer* l, matrix_t* x);

// hidden layer neurons forward
static int linear_backward(layer* l, matrix_t* grad);
// activation layer forward
static int relu_backward(layer* l, matrix_t* grad);
static int sigmoid_backward(layer* l, matrix_t* grad);
static int softmax_backward(layer* l, matrix_t* grad);


int forward(layer* l, matrix_t* x) {
  switch (l->type)
  {
    case relu:
      return relu_forward(l, x);
    case linear:
      return linear_forward(l, x);
    case sigmoid:
      return sigmoid_forward(l, x);
    default:
      printf("[FORWARD] Encountered unrecognized layer");
      exit(1);
  }
}

int backward(layer* l, matrix_t* grad) {
  switch (l->type)
  {
    case relu:
      return relu_backward(l, grad);
    case linear:
      return linear_backward(l, grad);
    case sigmoid:
      return sigmoid_backward(l, grad);
    default:
      printf("[BACKWARD] Encountered unrecognize layer");
      exit(1);
  }
}

static int relu_forward(layer* l, matrix_t* x) {
  relu_layer layer_data = l->data.r;
  int data_size = x->rows * x->cols;
  memcpy(layer_data.cache->data, x->data, data_size * sizeof(double));
  for (int i = 0; i < data_size; ++i) {
    if (x->data[i] < 0) {
      x->data[i] = layer_data.cache->data[i] = 0;
    }
    if (x->data[i] > 0) {
      layer_data.cache->data[i] = 1;
    }
  }
  return 1;
}

static int sigmoid_forward(layer* l, matrix_t* x) {
  sigmoid_layer layer_data = l->data.s;
  int data_size = x->rows * x->cols;
  for (int i = 0; i < data_size; ++i) x->data[i] = 1 / (1 + exp(-x->data[i]));
  memcpy(layer_data.cache->data, x->data, data_size * sizeof(double));
  return 1;
}

static int linear_forward(layer* l, matrix_t* x) {
  //caveat: address pointed by x should have enough space to hold new data
  linear_layer layer_data = l->data.l;
  int data_size = x->rows * x->cols;
  memcpy(layer_data.cache->data, x->data, data_size * sizeof(double));
  matrix_t* wx = matmul(x, layer_data.W);
  add_bias(wx, layer_data.b);
  
  //update the data flowing through the network
  memcpy(x->data, wx->data, wx->cols * wx->rows * sizeof(double));
  x->rows = wx->rows;
  x->cols = wx->cols;

  return 1;
}

static int softmax_forward(layer* l, matrix_t* x) {
  return 1;
}

static int linear_backward(layer* l, matrix_t* grad) {
  // caveat: memory needs to be realloced to hold new data
  linear_layer layer_data = l->data.l;

  matrix_t* cache_T = transpose(layer_data.cache);
  l->data.l.grad_W = matmul(cache_T, grad);
  matrix_t ones;
  ones.cols = grad->rows;
  ones.rows = 1;
  double ones_data[grad->rows];
  ones.data = ones_data;
  l->data.l.grad_b = matmul(&ones, grad);
  
  matrix_t* w_T = transpose(layer_data.W);
  matrix_t* new_grad = matmul(grad, w_T);
  memcpy(grad->data, new_grad->data, new_grad->cols*new_grad->rows*sizeof(double));
  grad->rows = new_grad->rows;
  grad->cols = new_grad->cols;

  return 1;
}

static int relu_backward(layer* l, matrix_t* grad) {
  elem_wise_mult(grad, l->data.r.cache);
  return 1;
}

static int sigmoid_backward(layer* l, matrix_t* grad) {
  matrix_t temp;
  sigmoid_layer layer_data = l->data.s;
  int size = layer_data.cache->cols * layer_data.cache->rows * sizeof(double)
              + 2 * sizeof(int);
  memcpy(&temp, layer_data.cache, size);
  neg(&temp);
  add_scalar(&temp, 1);
  elem_wise_mult(&temp, layer_data.cache);
  elem_wise_mult(grad, &temp);
  return 1;
}

static int softmax_backward(layer* l, matrix_t* grad) {
  return 1;
}
