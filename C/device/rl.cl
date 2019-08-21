__kernel void vector_add(__global const float* restrict x, 
                         __global const float* restrict y, 
                         __global float * restrict z)
{
  // get index of the work item
  int index = get_global_id(0);

  // add the vector elements
  z[index] = x[index] + y[index];
}

__kernel void gemm(__global const float* restrict x, const int x_row_, const int x_col_,
                    __global const float* restrict  y, const int y_row_, const int y_col_,
                    __global float* restrict r)
{
    

  __local int x_row; x_row = 100;
  __local int x_col; x_col = 100;
  __local int y_row; y_row = 100;
  __local int y_col; y_col = 100;
  __local int r_row; r_row = 100;
  __local int r_col; r_col = 100;
  
  __local int x_size; x_size = x_row*x_col;
  __local int y_size; y_size = y_row*y_col;
  __local int r_size; r_size = x_row*y_col;
  __local float cache_x[10000];
  __local float cache_y[10000];
  __local float cache_r[10000];

  for (size_t i = 0; i < x_row*x_col; ++i) cache_x[i] = x[i];
  for (size_t i = 0; i < y_row*y_col; ++i) cache_y[i] = y[i];

  // #pragma unroll 10
  for (size_t i = 0; i < r_row; ++i) {
    for (size_t k = 0; k < x_col; ++k) {
      // #pragma unroll
      for (size_t j = 0; j < r_col; ++j) {
        cache_r[i*r_col+j] += cache_x[i*x_col+k] * cache_y[k*y_col+j];
      }
    }
  }
  for (size_t i = 0; i < r_row*r_col; ++i) r[i] = cache_r[i]; 
}

void matmul(__global const float* restrict x,
            const int x_row,
            const int x_col,
            const int x_offset,
            __global const float* restrict y,
            const int y_row,
            const int y_col,
            const int y_offset,
            __global float* restrict r,
            __global int* restrict r_row,
            __global int* restrict r_col,
            const int r_offset) {
  int l_r_row; l_r_row = x_row;
  int l_r_col; l_r_col = y_col;
  // printf("[matmul] xr %d xc %d yr %d yc %d rr %d rc %d\n", x_row, x_col, y_row, y_col, l_r_row, l_r_col);
  for (int i = 0; i < l_r_row; ++i) {
    for (int k = 0; k < x_col; ++k) {
      for (int j = 0; j < l_r_col; ++j) {
        r[i*l_r_col+j+r_offset] += x[i*x_col+k+x_offset] * y[k*y_col+j+y_offset];
      }
    }
  }
  *r_row = l_r_row;
  *r_col = l_r_col;
}

void add_bias(__global float* restrict x,
              const int x_row,
              const int x_col,
              __global const float* restrict params,
              const int param_offset) {
  for (int i = 0; i < x_row; ++i) {
    for (int j = 0; j < x_col; ++j) {
     x[i*x_col+j] += params[j+param_offset];
    }
  }
}

__kernel void linear_forward_prop(__global const float* restrict params,
                                  __global int* restrict layer_param_offset,
                                  __global const int* restrict dims,
                                  __global const float* restrict input_data,
                                  __global int* restrict input_r,
                                  __global int* restrict input_c,
                                  __global float* restrict cache,
                                  __global int* restrict cache_offset_,
                                  __global float* restrict output,
                                  __global int* restrict output_r,
                                  __global int* restrict output_c,
                                  const int layer_idx,
                                  __global int* restrict err_code) {

  // local int layer_param_offset = 0;
  int cache_offset = *cache_offset_;
  // for (int i = 0; i < num_layers; i++) {
  int dim_offset = layer_idx*4;
  int W_r; W_r = dims[dim_offset];
  int W_c; W_c = dims[dim_offset+1];
  int b_r; b_r = dims[dim_offset+2];
  int b_c; b_c = dims[dim_offset+3];

  // clear output buffer
  for (int i = 0 ; i < *input_r*(W_c); ++i) {
    output[i] = 0;
  }

  // printf("[linear in] wr %d wc %d br %d bc %d ir %d ic %d p_offset %d c_offset %d l_idx %d\n", W_r, W_c, b_r, b_c, *input_r, *input_c, *layer_param_offset, *cache_offset_, layer_idx);

  // save cache
  for (int i = 0; i < *input_r*(*input_c); ++i) cache[cache_offset+i] = input_data[i];
  printf("cache offset from %d to %d\n", *cache_offset_, *cache_offset_+(*input_c)*(*input_r));
  *cache_offset_ += *input_r * *(input_c);

  // linear
  matmul(input_data, *input_r, *input_c, 0, params, W_r, W_c, *layer_param_offset, output, output_r, output_c, 0);
  // printf("------W------\n");
  // printf("starting from %d\n", *layer_param_offset);
  // for (int i = 0; i < 30; ++i) {
  //   printf("%e  ", params[i+*layer_param_offset]);
  // }
  // printf("\n");
  // printf("=============\n");
  // printf("------In-----\n");
  // for(int i = 0; i < 30; ++i) {
  //   printf("%e  ", input_data[i]);
  // }
  // printf("\n");
  // printf("=============\n");
  // printf("------O------\n");
  // for (int i = 0; i < 30; ++i) {
  //   printf("%e  ",  output[i]);
  // }
  // printf("\n");
  // printf("======O======\n");
  *layer_param_offset += W_r * W_c;
  add_bias(output, *output_r, *output_c, params, *layer_param_offset);
  // printf("------Ob------\n");
  // for (int i = 0; i < 30; ++i) {
  //   printf("%e  ",  output[i]);
  // }
  // printf("\n");
  // printf("======Ob======\n");
  *layer_param_offset += b_r * b_c;
  
  // printf("[linear out] or %d oc %d p_offset %d c_offset %d\n", *output_r, *output_c, *layer_param_offset, *cache_offset_);
  // }
}

__kernel void relu_forward_prop(__global float* restrict input_data,
                                __global int* restrict input_r,
                                __global int* restrict input_c,
                                __global float* restrict cache,
                                __global int* restrict cache_offset,
                                __global int* restrict err_code) {
  // printf("[relu] ir %d ic %d c_offset %d\n", *input_r, *input_c, *cache_offset);
  for (int i = 0; i < *input_c*(*input_r); ++i) {
    cache[*cache_offset+i] = input_data[i] > 0 ? 1 : 0;
    input_data[i] = input_data[i] > 0 ? input_data[i] : 0;
  }
  *cache_offset += *input_r * *input_c;
}

__kernel void mse(__global   const float* restrict input_data,
                  __global   const int* restrict input_r,
                  __global   const int* restrict input_c,
                  __global   float* restrict target_data,
                  __global   float* restrict tmp_buffer,
                  __global   float* restrict loss,
                  __global   int* restrict err_code) {

  int ir; ir = *input_r;
  int ic; ic = *input_c;
  for (int i = 0; i < ir*ic; ++i) {
    target_data[i] = input_data[i] - target_data[i];
    tmp_buffer[i] = pow(target_data[i], 2);
    target_data[i] *= (2.0 / (float)ir);
  }
  local float sum;
  sum = 0;
  
  for (int i = 0; i < *input_r*(*input_c); ++i) sum += tmp_buffer[i];
  *loss = sum / (*input_c * *input_r);
}

__kernel void relu_backward_prop(__global float* restrict input_grad,
                                 __global int* restrict input_grad_r,
                                 __global int* restrict input_grad_c,
                                 __global float* restrict cache,
                                 __global int* restrict cache_offset,
                                 __global int* restrict err_code) {
  printf("[relu in] offset at %d, changing to %d\n", *cache_offset, *cache_offset-*input_grad_r*(*input_grad_c));
  local int c_offset;
  *cache_offset -= *input_grad_r * (*input_grad_c);
  c_offset = *cache_offset;
  for (int i = 0; i < *input_grad_r*(*input_grad_c); ++i) input_grad[i] *= cache[c_offset+i];
}

__kernel void transpose_params_n_cache(__global float* restrict params,
                               __global float* restrict cache,
                               int num_layers,
                               int batch_size,
                               __global int* restrict dims,
                               __global float* restrict params_T,
                               __global float* restrict cache_T,
                               __global int* restrict layer_param_offset,
                               __global int* restrict err_code) {
  int param_offset; param_offset = 0;
  int cache_offset; cache_offset = 0;
  int param_origin_offset; param_origin_offset = 0;
  int c_r; c_r = batch_size;
  for (int n = 0; n < num_layers; ++n) {
    int dim_offset; dim_offset = n*4;
    // W
    int W_r; W_r = dims[dim_offset];
    int W_c; W_c = dims[dim_offset+1];
    for (int i = 0; i < W_c; ++i) {
      for (int j = 0; j < W_r; ++j) {
        params_T[i*W_r+j+param_offset] = params[j*W_c+i+param_origin_offset];
      }
    }
    param_origin_offset += W_r * W_c;
    param_offset += W_r * W_c;
    // b
    int b_r; b_r = dims[dim_offset+2];
    int b_c; b_c = dims[dim_offset+3];
    for (int i = 0; i < batch_size+b_r*b_c; ++i) {
      params_T[i+param_offset] = 1;
    }
    param_origin_offset += b_c*b_r;
    param_offset += batch_size + b_c*b_r;
    // cache_W
    int c_c; c_c = W_r;
    // printf("c_W cc %d cr %d\n", c_c, c_r);
    for (int i = 0; i < c_c; ++i) {
      for (int j = 0; j < c_r; ++j) {
        // printf("moving from %d: %f to %d: %f j %d i %d\n", j*c_c+i+cache_offset, cache[j*c_c+i+cache_offset], i*c_r+j+cache_offset,cache_T[i*c_r+j+cache_offset], j, i);
        cache_T[i*c_r+j+cache_offset] = cache[j*c_c+i+cache_offset];
      }
    }
    cache_offset += c_c*c_r;
    // cache_act
    c_c = W_c;
    // printf("c_A cc %d cr %d\n", c_c, c_r);
    for (int i = 0; i < c_c*c_r; ++i) {
      cache_T[i+cache_offset] = cache[i+cache_offset];
    }
    cache_offset += c_c*c_r;
  }
  // printf("param_global has value: %d\n", *layer_param_offset);
  // printf("setting param offset to %d\n", param_offset);
  *layer_param_offset = param_offset;
  // printf("param offset set to %d\n", *layer_param_offset);
}

__kernel void linear_backward_prop(__global const float* restrict params_T,
                                   __global int* restrict params_T_offset,
                                   __global int* restrict layer_param_offset,
                                   __global const int* restrict dims,
                                   __global const float* restrict input_grad,
                                   __global int* restrict input_grad_r,
                                   __global int* restrict input_grad_c,
                                   __global float* restrict cache_T,
                                   __global int* restrict cache_offset,
                                   __global float* restrict output_grad,
                                   __global int* restrict output_grad_r,
                                   __global int* restrict output_grad_c,
                                   __global float* restrict param_grads,
                                   int layer_idx,
                                   __global int* restrict err_code) {
  int layer_offset = layer_idx*4;
  int W_r; W_r = dims[layer_offset];
  int W_c; W_c = dims[layer_offset+1];
  int b_r; b_r = dims[layer_offset+2];
  int b_c; b_c = dims[layer_offset+3];
  // clear output buffer
  for (int i = 0; i < *input_grad_r*W_r; ++i) {
    output_grad[i] = 0;
  }
  // printf("[linear backward] idx %d ir %d ic %d\n", layer_idx, *input_grad_r, *input_grad_c);
  // printf("[linear in b] param offset at %d, changing to %d\n", *params_T_offset, *params_T_offset-*input_grad_r);

  *params_T_offset -= *input_grad_r;
  *layer_param_offset -= b_c*b_r;
  // for (int i= *layer_param_offset; i < *layer_param_offset+*input_grad_r; ++i) printf("%e ", params_T[i]);
  // printf("\n");
  matmul(params_T, 1, *input_grad_r, *params_T_offset, input_grad, *input_grad_r, *input_grad_c, 0, param_grads, output_grad_r, output_grad_c, *layer_param_offset);
  // int W_offset = *layer_param_offset;
  *params_T_offset -= b_c*b_r;
  // printf("[linear in W] param offset at %d, changing to %d\n", *layer_param_offset, *layer_param_offset-W_r*W_c);
  // printf("[linear in C] cache offset at %d, changing to %d\n", *cache_offset, *cache_offset-W_r*(*input_grad_r));

  *layer_param_offset -= W_r * W_c;
  *params_T_offset -= W_r * W_c;
  *cache_offset -= W_r * (*input_grad_r);
  matmul(cache_T, W_r, *input_grad_r, *cache_offset, input_grad, *input_grad_r, *input_grad_c, 0, param_grads, output_grad_r, output_grad_c, *layer_param_offset);
  matmul(input_grad, *input_grad_r, *input_grad_c, 0, params_T, W_c, W_r, *params_T_offset, output_grad, output_grad_r, output_grad_c, 0);
  // printf("device grads\n");
  // printf("In----------\n");
  // for (int i = 0; i < *input_grad_c*(*input_grad_r); ++i) printf("%e  ", input_grad[i]);
  // printf("\n");
  // printf("param_T\n");
  // for (int i = *params_T_offset; i < *params_T_offset+W_c*W_r; ++i) printf("%e  ", params_T[i]);
  // printf("\n");
  // printf("device grads end------------\n");
}

__kernel void generate_update_adam(__global float* restrict params,
                                   __global float* restrict first_moment,
                                   __global float* restrict second_moment,
                                   __global float* restrict grads,
                                   __global int* restrict grad_size,
                                   __global int* restrict timestamp,
                                   __global float* restrict beta1,
                                   __global float* restrict beta2,
                                   __global float* restrict epsilon,
                                   __global float* restrict learning_rate,
                                   __global int* restrict err_code) {
  *timestamp += 1;
  float beta1_exp; beta1_exp = pow(*beta1, *timestamp);
  float beta2_exp; beta2_exp = pow(*beta2, *timestamp);
  float corrected_fst;
  float corrected_snd;
  float lr = *learning_rate * (sqrt(1-beta2_exp)/(1-beta1_exp));
  for (int i = 0; i < *grad_size; ++i) {
    first_moment[i] = *beta1 * (first_moment[i] - grads[i]) + grads[i];
    second_moment[i] = *beta2 * (second_moment[i] - grads[i]) + pow(grads[i], 2);
    corrected_snd = sqrt(second_moment[i]) + *epsilon;
    corrected_fst = (first_moment[i] / corrected_snd) * lr;
    params[i] -= corrected_fst;
  }
}

__kernel void examine_float_array(__global float* restrict params,
                             const int param_size,
                             int last_n_digit) {
  if (last_n_digit == -1) last_n_digit = param_size;
  // for (int i = param_size - last_n_digit; i < param_size; ++i) {
  for (int i = 0; i < last_n_digit; ++i) {
    printf("%e ", params[i]);
  }
  printf("\n");
}

__kernel void examine_int_array(__global int* restrict arr,
                                const int param_size,
                                int last_n_digit) {
  if (last_n_digit == -1) last_n_digit = param_size;
  for (int i = param_size - last_n_digit; i < param_size; ++i) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}
