#include "layers.h"
#include <stdlib.h>
#include <stdio.h>
#include "matrix_op.h"
#include <string.h>
#include <math.h>
#include <assert.h>

// hidden layer neurons forward
static int linear_forward(layer* l, matrix_t* x);
// activation layer forward
static int relu_forward(layer* l, matrix_t* x);
static int sigmoid_forward(layer* l, matrix_t* x);
static int placeholder_forward(layer* l, matrix_t* x) {return 1;}
static int tanh_forward(layer* l, matrix_t* x);
// static int softmax_forward(layer* l, matrix_t* x);

// hidden layer neurons backward
static int linear_backward(layer* l, matrix_t* grad);
// activation layer backward
static int relu_backward(layer* l, matrix_t* grad);
static int sigmoid_backward(layer* l, matrix_t* grad);
static int placeholder_backward(layer* l, matrix_t* x) {return 1;}
static int tanh_backward(layer* l, matrix_t* grad);
// static int softmax_backward(layer* l, matrix_t* grad);

// hidden layer weights update
static int linear_update(layer* l, double learning_rate);

// loss layer forward and backward
static double mse_loss_forward(layer* l, matrix_t* x, matrix_t* target);
static matrix_t* mse_loss_backward(layer* l);

int forward(layer* l, matrix_t* x) {
  switch (l->type) {
    case relu:
      return relu_forward(l, x);
    case linear:
      return linear_forward(l, x);
    case sigmoid:
      return sigmoid_forward(l, x);
    case tanh_:
      return tanh_forward(l, x);
    case placeholder:
      return placeholder_forward(l, x);
    default:
      printf("[HIDDEN FORWARD] Encountered unrecognized layer, %d", l->type);
      exit(1);
  }
}

int backward(layer* l, matrix_t* grad) {
  switch (l->type) {
    case relu:
      return relu_backward(l, grad);
    case linear:
      return linear_backward(l, grad);
    case sigmoid:
      return sigmoid_backward(l, grad);
    case tanh_:
      return tanh_backward(l, grad);
    case placeholder:
      return placeholder_backward(l, grad);
    default:
      printf("[HIDDEN BACKWARD] Encountered unrecognize layer, %d", l->type);
      exit(1);
  }
}

int update(layer* l, double learning_rate) {
  switch (l->type) {
    case linear:
      return linear_update(l, learning_rate);
    default:
      break;
  }
  return 1;
}

double loss_forward(layer* l, matrix_t* x, matrix_t* target) {
  switch (l->type) {
    case mse_loss:
      return mse_loss_forward(l, x, target);
    default:
      printf("[LOSS FORWARD] Encountered unrecognized layer, %d", l->type);
      exit(1);
  }
  return 1;
}

matrix_t* loss_backward(layer* l) {
  switch (l->type) {
    case mse_loss:
      return mse_loss_backward(l);
    default:
      printf("[LOSS BACKWARD] Encountered unrecognized layer, %d", l->type);
      exit(1);
  }
}

int linear_update(layer* l, double learning_rate) {
  assert(l->type == linear);
  linear_layer layer_data = l->data.l;

  mult_scalar(layer_data.grad_b, learning_rate);
  mult_scalar(layer_data.grad_W, learning_rate);

  elem_wise_minus(layer_data.W, layer_data.grad_W);
  elem_wise_minus(layer_data.b, layer_data.grad_b);

  return 1;
}

static int relu_forward(layer* l, matrix_t* x) {
  relu_layer layer_data = l->data.r;
  int data_size = x->rows * x->cols;
  copy_matrix(layer_data.cache, x);
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
  copy_matrix(layer_data.cache, x);
  return 1;
}

static int linear_forward(layer* l, matrix_t* x) {
  //caveat: address pointed by x should have enough space to hold new data
  linear_layer layer_data = l->data.l;

  copy_matrix(layer_data.cache, x);
  matrix_t* wx = matmul(x, layer_data.W);
  add_bias(wx, layer_data.b);
  
  //update the data flowing through the network
  copy_matrix(x, wx);
  if (contains_nan(x)) {
    printf("[LINEAR FORWARD]");
    // print_matrix(x,1);
    // print_matrix(layer_data.W, 1);
    exit(1);
  }
  free_matrix(wx);
  return 1;
}

// static int softmax_forward(layer* l, matrix_t* x) {
//   //softmax_layer layer_data = l->data.so;

//   return 1;
// }

static int linear_backward(layer* l, matrix_t* grad) {
  // caveat: memory needs to be realloced to hold new data
  linear_layer layer_data = l->data.l;
  free_matrix(l->data.l.grad_W);
  free_matrix(l->data.l.grad_b);

  matrix_t* cache_T = transpose(layer_data.cache);

  l->data.l.grad_W = matmul(cache_T, grad);

  matrix_t* ones = new_matrix(1, grad->rows);
  for (int i = 0; i < grad->rows; ++i) {
    ones->data[i] = 1;
  }
  l->data.l.grad_b = matmul(ones, grad);
  
  free_matrix(ones);
  
  matrix_t* w_T = transpose(layer_data.W);
  matrix_t* new_grad = matmul(grad, w_T);

  copy_matrix(grad, new_grad);
  free_matrix(cache_T);
  free_matrix(w_T);
  free_matrix(new_grad);
  // printf("[normal===========================]\n");
  // print_matrix(l->data.l.grad_W, 1);
  // printf("[normal============================]\n");
  if (contains_nan(grad)) {
    printf("[LINEAR BACKWARD]");
    // print_matrix(grad,1);
    // print_matrix(layer_data.W, 1);
    exit(1);
  }
  return 1;
}

static int relu_backward(layer* l, matrix_t* grad) {
  elem_wise_mult(grad, l->data.r.cache);
  return 1;
}

static int sigmoid_backward(layer* l, matrix_t* grad) {
  sigmoid_layer layer_data = l->data.s;
  matrix_t* temp = new_matrix(layer_data.cache->rows, layer_data.cache->cols);

  copy_matrix(temp, layer_data.cache);
  neg(temp);
  add_scalar(temp, 1);
  elem_wise_mult(temp, layer_data.cache);

  elem_wise_mult(grad, temp);
  free_matrix(temp);

  return 1;
}

static int tanh_forward(layer* l, matrix_t* x) {
  // apply activation
  mult_scalar(x, 2);
  neg(x);
  for (int i = 0; i < x->rows*x->cols; ++i) x->data[i] = exp(x->data[i]);
  add_scalar(x, 1);
  inverse(x);
  mult_scalar(x, 2);
  add_scalar(x, -1);
  // keep gradient
  copy_matrix(l->data.t.cache, x);
  elem_wise_mult(l->data.t.cache, l->data.t.cache);
  neg(l->data.t.cache);
  add_scalar(l->data.t.cache, 1);
  return 1;
}

static int tanh_backward(layer* l, matrix_t* grad) {
  elem_wise_mult(grad, l->data.t.cache);
  return 1;
}

// static int softmax_backward(layer* l, matrix_t* grad) {
//   return 1;
// }

static double mse_loss_forward(layer* l, matrix_t* x, matrix_t* target) {
  mse_loss_layer layer_data = l->data.m;
  copy_matrix(layer_data.cache_pred, x);
  copy_matrix(layer_data.cache_target, target);
  elem_wise_minus(x, target);
  elem_wise_mult(x, x);
  double loss = mean(x);
  free_matrix(x);
  free_matrix(target);
  return loss;
}

static matrix_t* mse_loss_backward(layer* l) {
  mse_loss_layer layer_data = l->data.m;
  int cols = layer_data.cache_target->cols;
  int rows = layer_data.cache_target->rows;
  matrix_t* grad = new_matrix(rows, cols);
  copy_matrix(grad, layer_data.cache_pred);
  elem_wise_minus(grad, layer_data.cache_target);
  mult_scalar(grad, (double)2);
  mult_scalar(grad, 1/((double)rows));
  return grad;
}

int init_linear(layer* l, int in, int out) {
  // new linear layer
  linear_layer new_linear_layer;
  // initialize member ptrs. Cache needs to be realloc at fit time
  new_linear_layer.W = new_matrix(in, out);
  new_linear_layer.b = new_matrix(1, out);
  initialize(new_linear_layer.W, xavier);
  initialize(new_linear_layer.b, xavier);
  new_linear_layer.grad_W = new_matrix(in, out);
  new_linear_layer.grad_b = new_matrix(1, out);
  // wrap up with struct layer
  l->type = linear;
  l->data.l = new_linear_layer;
  
  return 1;
}

int free_layer(layer l) {
  layer_data data = l.data;
  switch (l.type) {
    case relu: {
      free_matrix(data.r.cache);
      break;
    }
    case linear: {
      free_matrix(data.l.W);
      free_matrix(data.l.b);
      free_matrix(data.l.grad_b);
      free_matrix(data.l.grad_W);
      free_matrix(data.l.cache);
      break;
    }
    case sigmoid: {
      free_matrix(data.s.cache);
      break;
    }
    case tanh_: {
      free_matrix(data.t.cache);
      break;
    }
    case mse_loss: {
      free_matrix(data.m.cache_pred);
      free_matrix(data.m.cache_target);
      break;
    }
    default:
      break;
  }
  return 1;
}
