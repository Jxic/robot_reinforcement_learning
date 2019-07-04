#include "layers.h"
#include <stdlib.h>
#include <stdio.h>
#include "matrix_op.h"
#include <string.h>
#include <math.h>
#include <assert.h>
#include "macros.h"

// hidden layer neurons forward
static int linear_forward(layer* l, matrix_t* x);
// activation layer forward
static int relu_forward(layer* l, matrix_t* x);
static int sigmoid_forward(layer* l, matrix_t* x);
static int placeholder_forward(layer* l, matrix_t* x) {return 1;}
static int tanh_forward(layer* l, matrix_t* x);

// hidden layer neurons backward
static int linear_backward(layer* l, matrix_t* grad);
static int conv_backward(layer* l, matrix_t* grad);
// activation layer backward
static int relu_backward(layer* l, matrix_t* grad);
static int sigmoid_backward(layer* l, matrix_t* grad);
static int placeholder_backward(layer* l, matrix_t* x) {return 1;}
static int tanh_backward(layer* l, matrix_t* grad);

// hidden layer weights update
static int linear_update(layer* l, double learning_rate);
static int conv_update(layer* l, double learning_rate);

// loss layer forward and backward
static double mse_loss_forward(layer* l, matrix_t* x, matrix_t* target);
static matrix_t* mse_loss_backward(layer* l);
static double cross_entropy_forward(layer* l, matrix_t* x, matrix_t* target);
static matrix_t* cross_entropy_backward(layer* l);

// helper
//  static int conv_reconstruct_output(matrix_t* output, int o_rows, int o_cols, int o_channels, matrix_t* out);

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
    case conv:
      return conv_forward(l, x);
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
    case conv:
      return conv_backward(l, grad);
    default:
      printf("[HIDDEN BACKWARD] Encountered unrecognize layer, %d", l->type);
      exit(1);
  }
}

int update(layer* l, double learning_rate) {
  switch (l->type) {
    case linear:
      return linear_update(l, learning_rate);
    case conv:
      return conv_update(l, learning_rate);
    default:
      break;
  }
  return 1;
}

double loss_forward(layer* l, matrix_t* x, matrix_t* target) {
  switch (l->type) {
    case mse_loss:
      return mse_loss_forward(l, x, target);
    case cce_loss:
      return cross_entropy_forward(l, x, target);
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
    case cce_loss:
      return cross_entropy_backward(l);
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

int conv_update(layer* l, double learning_rate) {
  assert(l->type == conv);
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

  matrix_t* wx = new_matrix(x->rows, layer_data.W->cols);
  matmul(x, layer_data.W, wx);
  add_bias(wx, layer_data.b);
  
  //update the data flowing through the network
  copy_matrix(x, wx);
  // if (contains_nan(x)) {
  //   printf("[LINEAR FORWARD]");
  //   // print_matrix(x,1);
  //   // print_matrix(layer_data.W, 1);
  //   exit(1);
  // }
  free_matrix(wx);
  return 1;
}

static int linear_backward(layer* l, matrix_t* grad) {
  // caveat: memory needs to be realloced to hold new data
  linear_layer layer_data = l->data.l;
  matrix_t* cache_T = transpose(layer_data.cache);
  matrix_t* ones = new_matrix(1, grad->rows);
  for (int i = 0; i < grad->rows; ++i) {
    ones->data[i] = 1;
  }
  matrix_t* w_T = transpose(layer_data.W);
  #ifdef GPU
  if (layer_data.W->rows*layer_data.W->cols >= 4000000000000000000) {
    matrix_t** updates = mat_mul_series(cache_T, grad, ones, grad, grad, w_T);

    copy_matrix(l->data.l.grad_W, updates[0]);
    copy_matrix(l->data.l.grad_b, updates[1]);
    free_matrix(updates[0]);
    free_matrix(updates[1]);
    
    copy_matrix(grad, updates[2]);
    free_matrix(updates[2]);
    free(updates);
  } else {
  #endif
    matrix_t* new_grad = new_matrix(grad->rows, w_T->cols);
    matmul(cache_T, grad, l->data.l.grad_W);
    matmul(ones, grad, l->data.l.grad_b);
    matmul(grad, w_T, new_grad);
    copy_matrix(grad, new_grad);
    free_matrix(new_grad);
  #ifdef GPU
  }
  #endif
  free_matrix(ones);
  free_matrix(cache_T);
  free_matrix(w_T);
  // if (contains_nan(grad)) {
  //   printf("[LINEAR BACKWARD]");
  //   // print_matrix(grad,1);
  //   // print_matrix(layer_data.W, 1);
  //   exit(1);
  // }
  return 1;
}

int conv_forward(layer* l, matrix_t* x) {
  linear_layer layer_data = l->data.l;
  int f_nums = layer_data.W->cols;
  int f_rows = layer_data.sizes[0];
  int f_cols = layer_data.sizes[1];
  int f_channels = layer_data.sizes[2];
  assert(f_rows*f_cols*f_channels == layer_data.W->rows);
  int stride = layer_data.stride;
  int padding = layer_data.padding;
  int i_rows = layer_data.input_sizes[0];
  int i_cols = layer_data.input_sizes[1];
  int i_channels = layer_data.input_sizes[2];
  assert(i_rows*i_cols*i_channels == x->cols);
  int single_filter_output_dim = (i_rows + padding * 2 - f_rows) / stride + 1;
  int single_filter_output_size = pow(single_filter_output_dim, 2);

  int batch = x->rows;


  matrix_t* W = layer_data.W;
  matrix_t* b = layer_data.b;

  matrix_t* recon = conv_reconstruct_input(x, i_rows, i_cols, i_channels, f_rows, f_cols, f_channels, stride, padding);
  matmul(recon, W, x);
  print_matrix(x, 1);
  add_bias(x, b);

  matrix_t* trans_x = transpose(x);
  matrix_t* fst_row = slice_col_wise(trans_x, 0, single_filter_output_size);
  fst_row->cols *= fst_row->cols;
  fst_row->rows = 1;
  printf("trans\n");
  print_matrix(trans_x, 1);

  for (int i = 1; i < batch; ++i) {
    matrix_t* nxt_row = slice_col_wise(trans_x, i*single_filter_output_size, (i+1)*single_filter_output_size);
    nxt_row->cols = nxt_row->rows * nxt_row->cols;
    nxt_row->rows = 1;
    matrix_t* new = concatenate(fst_row, nxt_row, 0);
    free_matrix(fst_row);
    free_matrix(nxt_row);
    fst_row = new;
  }

  free_matrix(trans_x);
  copy_matrix(x, fst_row);
  // add_bias(x, b);
  free_matrix(fst_row);
  free_matrix(layer_data.cache);
  l->data.l.cache = recon;


  return 1;
}

static int conv_backward(layer* l, matrix_t* grad){
  matrix_t* recon = l->data.l.cache;
  matrix_t* grad_b = l->data.l.grad_b;
  matrix_t* grad_W = l->data.l.grad_W;
  int f_num = l->data.l.W->cols;

  // initialize(grad_W, zeros);
  int single_filter_output_size = grad->cols / f_num;
  matrix_t* fst_row = slice_col_wise(grad, 0, single_filter_output_size);
  matrix_t* sum = matrix_sum(fst_row, 2);
  grad_b->data[0] = sum->data[0];
  free_matrix(sum);
  fst_row->cols *= fst_row->rows;
  fst_row->rows = 1;

  for (int i = 1; i < f_num; ++i) {
    matrix_t* nxt_row = slice_col_wise(grad, i*single_filter_output_size, (i+1)*single_filter_output_size);
    sum = matrix_sum(nxt_row, 2);
    grad_b->data[i] = sum->data[0];
    free_matrix(sum);
    nxt_row->cols = nxt_row->rows * nxt_row->cols;
    nxt_row->rows = 1;
    matrix_t* new = concatenate(fst_row, nxt_row, 0);
    free_matrix(fst_row);
    free_matrix(nxt_row);
    fst_row = new;
  }

  matrix_t* new_grad = new_matrix(f_num, grad_W->rows);
  matmul(fst_row, recon, new_grad);
  matrix_t* new_grad_T = transpose(new_grad);

  copy_matrix(grad_W, new_grad_T);

  // update grad x
  grad->rows *= grad->cols;
  grad->cols = 1;

  matrix_t* windows_grad_T = new_matrix(grad_W->rows, fst_row->cols);
  matrix_t* windows_grad = transpose(windows_grad_T);
  update_grad_x(windows_grad, grad, l);

  return 1;
}

int update_grad_x(matrix_t* w, matrix_t* x, layer* l) {
  linear_layer layer_data = l->data.l;
  int f_nums = layer_data.W->cols;
  int f_rows = layer_data.sizes[0];
  int f_cols = layer_data.sizes[1];
  int f_channels = layer_data.sizes[2];
  assert(f_rows*f_cols*f_channels == layer_data.W->rows);
  int stride = layer_data.stride;
  int padding = layer_data.padding;
  int i_rows = layer_data.input_sizes[0];
  int i_cols = layer_data.input_sizes[1];
  int i_channels = layer_data.input_sizes[2];
  assert(i_rows*i_cols*i_channels == x->cols);
  int single_filter_output_dim = (i_rows + padding * 2 - f_rows) / stride + 1;
  int single_filter_output_size = pow(single_filter_output_dim, 2);

  int batch = x->rows;

  for (int i = 0; i < batch; ++i) {
    for (int k = 0; k < i_channels; ++k) {
      for (int r = 0; r < i_rows; ++r) {
        for (int c = 0; c < i_cols; ++c) {
          
        }
      }
    }
  }

  return 1;
}



matrix_t* conv_reconstruct_input(matrix_t* input, int i_rows, int i_cols, int i_channels, int f_rows, int f_cols, int f_channels, int stride, int padding) {
  assert(i_rows == i_cols); // assuming squared input for now, assuming input shape divisible by f_rows f_cols for now
  printf("reconstructing batch: %d (ir: %d, ic: %d, ich: %d) (fr: %d, fc: %d, fch: %d) stride: %d padding %d\n", input->rows,i_rows, i_cols, i_channels, f_rows, f_cols, f_channels, stride, padding);
  matrix_t* ret;
  int single_channel_size = (i_rows + padding * 2 - f_rows) / stride + 1;
  printf("single channel size %d\n", single_channel_size);
  int single_output_size = pow(single_channel_size,2);
  printf("reconstructed rows %d\n",single_output_size*input->rows);
  ret = new_matrix(single_output_size*input->rows, f_rows*f_cols*f_channels);

  matrix_t* prep_input = padding ? padded_input(input, i_rows, i_cols, i_channels, padding) : matrix_clone(input);
  printf("padded dimension %d %d\n", prep_input->rows, prep_input->cols);
  print_matrix(prep_input, 1);
  int sub_cube_count = 0;
  for (int i = 0; i < prep_input->rows; ++i) {
    for (int r = 0; r < i_rows+padding*2; r += stride) {
      if (r + f_cols > i_cols+padding*2) continue;
      for (int c = 0; c < i_cols+padding*2; c += stride) {
        if (c + f_rows > i_rows+padding*2) continue;
        for (int j = 0; j < i_channels; ++j) {
          for (int sub_r = 0; sub_r < f_rows; ++sub_r) {
            // printf("moving from sample: %d, row: %d, col: %d, channel: %d, subrow: %d\n", i, r, c, j, sub_r);
            double* from = prep_input->data + i*prep_input->cols + j*(i_cols+padding*2)*(i_rows+padding*2) + r*(i_cols+padding*2) + c + sub_r*(i_cols+padding*2);
          
            double* to = ret->data + sub_cube_count*ret->cols + j*f_rows*f_cols + sub_r*f_cols;
            
            memcpy(to, from, f_cols*sizeof(double));
            if (i) {
              printf("moving %f %f %f to %d row\n", *from, *(from+1), *(from+2), sub_cube_count);
            }
          }
        }
        sub_cube_count++;
      }
    }
  }

  return ret;
}

static int conv_reconstruct_output(matrix_t* output, int o_rows, int o_cols, int o_channels, matrix_t* out) {
  return 1;
}

matrix_t* padded_input(matrix_t* input, int i_rows, int i_cols, int i_channels, int padding) {
  matrix_t* ret;

  int single_channel_size = (i_rows + padding*2) * (i_cols+padding*2);
  int padded_size = single_channel_size * i_channels;
  
  ret = new_matrix(input->rows, padded_size);
  initialize(ret, zeros);
  for (int i = 0; i < input->rows; ++i) {
    for (int k = 0; k < i_channels; ++k) {
      int row_pos = 0;
      for (int j = (i_cols+padding*2)*padding+padding+k*single_channel_size; j < single_channel_size+k*single_channel_size-((i_cols+padding*2)*padding+padding); j += i_cols+padding*2) {
        double* from = input->data+i*input->cols+k*i_cols*i_rows+row_pos*i_cols;
        double* to = ret->data+i*ret->cols+j;
        printf("j %d k %d i %d row_pos %d leading %f\n", j, k, i, row_pos, *(from));
        memcpy(to, from, i_cols*sizeof(double));
        row_pos++;
      }
    }
  }

  return ret;
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
  // for (int i = 0; i < x->rows*x->cols; ++i) x->data[i] = exp(x->data[i]);
  matrix_exp(x);
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
  mult_scalar(grad, 2/(double)rows);
  //mult_scalar(grad, 1/((double)rows));
  return grad;
}

static double cross_entropy_forward(layer* l, matrix_t* x, matrix_t* target) {
  cce_loss_layer layer_data = l->data.c;
  // softmax
  
  matrix_exp(x);
  matrix_t* sums = matrix_sum(x, 1);
  for (int i = 0; i < x->rows; ++i) {
    for (int j = 0; j < x->cols; ++j) {
      x->data[i*x->cols+j] /= sums->data[i];
    }
  }
  copy_matrix(layer_data.cache_pred, x);
  copy_matrix(layer_data.cache_target, target);

  // cross entropy

  matrix_log(x);


  elem_wise_mult(x, target);
  
  matrix_t* loss_sum = matrix_sum(x,2);
  double loss = loss_sum->data[0];
  loss /= -(double)x->rows;

  // printf("loss %f\n", loss);

  free_matrix(x);
  free_matrix(target);
  free_matrix(loss_sum);

  return loss;
}

static matrix_t* cross_entropy_backward(layer* l) {
  cce_loss_layer layer_data = l->data.c;
  int rows = layer_data.cache_target->rows;
  int cols = layer_data.cache_target->cols;
  matrix_t* grad = new_matrix(rows, cols);
  copy_matrix(grad, layer_data.cache_target);
  elem_wise_minus(grad, layer_data.cache_pred);
  mult_scalar(grad, -1/(double)rows);
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
