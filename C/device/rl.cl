#include "../macros.h"

#ifdef USING_CHANNEL
#pragma OPENCL EXTENSION cl_intel_channels : enable
channel int l_token[10] __attribute__ ((depth(10)));
channel int l_ack[10] __attribute__ ((depth(10)));
channel int a_token[10] __attribute__ ((depth(10)));

channel float c_li[10] __attribute__ ((depth(10)));
channel float c_ri[10] __attribute__ ((depth(10)));
#endif

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

#ifdef USING_CHANNEL
void matmul_w_bias_from_channel(__global int* restrict x_row,
            __global int* restrict x_col,
            const int x_offset,
            __global const float* restrict y,
            const int y_row,
            const int y_col,
            const int y_offset,
            __global float* restrict r,
            __global int* restrict r_row,
            __global int* restrict r_col,
            const int r_offset,
            const int bias_offset,
            const int layer_idx) {
  int l_r_row; l_r_row = *x_row;
  int l_r_col; l_r_col = y_col;
  int x_row_ = *x_row;
  int x_col_ = *x_col;
  *r_row = l_r_row;
  *r_col = l_r_col;
  *x_row = l_r_row;
  *x_col = l_r_col;
  printf("layer %d allowing for activation to start, or %d oc %d\n", layer_idx, *r_row, *r_col);
  // write_channel_intel(a_token[layer_idx], 1);
  switch (layer_idx) {
    case 0: write_channel_intel(a_token[0], 1); break;
    case 1: write_channel_intel(a_token[1], 1); break;
    case 2: write_channel_intel(a_token[2], 1); break;
    case 3: write_channel_intel(a_token[3], 1); break;
    case 4: write_channel_intel(a_token[4], 1); break;
    default: printf("Error!!!!!!\n");
  }
  // printf("[matmul] xr %d xc %d yr %d yc %d rr %d rc %d\n", x_row, x_col, y_row, y_col, l_r_row, l_r_col);
  float x[1024];
  float r_[1024];
  
  for (int i = 0; i < l_r_row; ++i) {
    for (int ri = 0; ri < l_r_col; ++ri) {
      r_[ri] = 0;
    }
    for (int vi = 0; vi < x_col_; ++vi) {
      // x[vi] = read_channel_intel(c_li[layer_idx]);
      switch (layer_idx) {
        case 0: x[vi] = read_channel_intel(c_li[0]); break;
        case 1: x[vi] = read_channel_intel(c_li[1]); break;
        case 2: x[vi] = read_channel_intel(c_li[2]); break;
        case 3: x[vi] = read_channel_intel(c_li[3]); break;
        case 4: x[vi] = read_channel_intel(c_li[4]); break;
        default: printf("Error!!!!!!\n");
      }
    }
    for (int j = 0; j < l_r_col; ++j) {
      for (int k = 0; k < x_col_; ++k) {
        r_[j] += x[k] * y[k*y_col+j+y_offset];
      }
    }

    for (int b = 0; b < l_r_col; ++b) {
      // write_channel_intel(c_ri[layer_idx], r_[b] + y[bias_offset+b]);
      switch (layer_idx) {
        case 0: write_channel_intel(c_ri[0], r_[b] + y[bias_offset+b]); break;
        case 1: write_channel_intel(c_ri[1], r_[b] + y[bias_offset+b]); break;
        case 2: write_channel_intel(c_ri[2], r_[b] + y[bias_offset+b]); break;
        case 3: write_channel_intel(c_ri[3], r_[b] + y[bias_offset+b]); break;
        case 4: write_channel_intel(c_ri[4], r_[b] + y[bias_offset+b]); break;
        default: printf("Error!!!!!!\n");
      }
    }
  }
}
#endif

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

#ifdef USING_CHANNEL
__attribute__((max_global_work_dim(0)))
__kernel void channel_start(__global const float* restrict input_data,
                            __global int* restrict input_r,
                            __global int* restrict input_c) {
  for (int i = 0; i < *input_r*(*input_c); ++i) {
    write_channel_intel(c_li[0], input_data[i]);
  }
}

__kernel void channel_end(__global float* restrict buffer,
                          __global int* restrict output_r,
                          __global int* restrict output_c,
                          const int offset,
                          const int idx) {
  // read_channel_intel(a_token[idx]);
  switch (idx) {
    case 0: read_channel_intel(a_token[0]); break;
    case 1: read_channel_intel(a_token[1]); break;
    case 2: read_channel_intel(a_token[2]); break;
    case 3: read_channel_intel(a_token[3]); break;
    case 4: read_channel_intel(a_token[4]); break;
    default: printf("Error!!!!!!\n");
  }
  for (int i = 0; i < *output_r*(*output_c); ++i) {
    // buffer[offset+i] = read_channel_intel(c_ri[idx]);
    switch (idx) {
        case 0: buffer[offset+i] = read_channel_intel(c_ri[0]); break;
        case 1: buffer[offset+i] = read_channel_intel(c_ri[1]); break;
        case 2: buffer[offset+i] = read_channel_intel(c_ri[2]); break;
        case 3: buffer[offset+i] = read_channel_intel(c_ri[3]); break;
        case 4: buffer[offset+i] = read_channel_intel(c_ri[4]); break;
        default: printf("Error!!!!!!\n");
    }
  }
}

__attribute__((max_global_work_dim(0)))
__kernel void channel_manager(const int num_of_layers) {
  for (int i = 0; i < num_of_layers; ++i) {
    printf("allowing for forward propagation of layer %d/%d\n", i+1, num_of_layers);
    // write_channel_intel(l_token[i], 1);
    // read_channel_intel(l_ack[i]);
    switch (i) {
        case 0:  write_channel_intel(l_token[0], 1); read_channel_intel(l_ack[0]); break;
        case 1:  write_channel_intel(l_token[1], 1); read_channel_intel(l_ack[1]); break;
        case 2:  write_channel_intel(l_token[2], 1); read_channel_intel(l_ack[2]); break;
        case 3:  write_channel_intel(l_token[3], 1); read_channel_intel(l_ack[3]); break;
        case 4:  write_channel_intel(l_token[4], 1); read_channel_intel(l_ack[4]); break;
        default: printf("Error!!!!!!\n");
    }
    printf("got acknowledgement from layer %d\n", i);
  }
}
#endif

#ifdef USING_CHANNEL
__attribute__((num_compute_units(5)))
#endif
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
                                  int layer_idx,
                                  const int layer_idx_max,
                                  __global int* restrict err_code) {
  #ifdef USING_CHANNEL
  layer_idx = get_global_id(0);
  if (layer_idx > layer_idx_max) return;
  printf("layer %d trying to acquire token, max %d\n", layer_idx, layer_idx_max);
  // read_channel_intel(l_token[layer_idx]);
  switch (layer_idx) {
    case 0: read_channel_intel(l_token[0]); break;
    case 1: read_channel_intel(l_token[1]); break;
    case 2: read_channel_intel(l_token[2]); break;
    case 3: read_channel_intel(l_token[3]); break;
    case 4: read_channel_intel(l_token[4]); break;
    default: printf("Error!!!!!!\n");
  }
  printf("layer %d got token starting, input_r: %d, input_c: %d\n", layer_idx , *input_r, *input_c);
  #endif
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
  // save cache
  for (int i = 0; i < *input_r*(*input_c); ++i) cache[cache_offset+i] = input_data[i];
  *cache_offset_ += *input_r * *(input_c);

  #ifdef USING_CHANNEL
  int layer_param_offset_ = *layer_param_offset;
  *layer_param_offset += W_r * W_c + b_r * b_c;
  matmul_w_bias_from_channel(input_r, input_c, 0, params, W_r, W_c, layer_param_offset_, output, output_r, output_c, 0, layer_param_offset_ + W_r * W_c, layer_idx);
  #endif

  #ifndef USING_CHANNEL
  matmul(input_data, *input_r, *input_c, 0, params, W_r, W_c, *layer_param_offset, output, output_r, output_c, 0);
  *layer_param_offset += W_r * W_c;
  #endif
  #ifndef USING_CHANNEL
  add_bias(output, *output_r, *output_c, params, *layer_param_offset);
  *layer_param_offset += b_r * b_c;
  #endif
  // printf("layer %d finished\n", layer_idx);
}

#ifdef USING_CHANNEL
__attribute__((num_compute_units(5)))
#endif
__kernel void relu_forward_prop(__global float* restrict input_data,
                                __global int* restrict input_r,
                                __global int* restrict input_c,
                                __global float* restrict cache,
                                __global int* restrict cache_offset,
                                int layer_idx,
                                const int layer_idx_max,
                                __global int* restrict err_code) {
  #ifdef USING_CHANNEL
  layer_idx = get_global_id(0);
  if (layer_idx > layer_idx_max) return;
  printf("relu layer %d trying to acquire token, max %d\n", layer_idx, layer_idx_max);
  // read_channel_intel(a_token[layer_idx]);
  switch (layer_idx) {
    case 0: read_channel_intel(a_token[0]); break;
    case 1: read_channel_intel(a_token[1]); break;
    case 2: read_channel_intel(a_token[2]); break;
    case 3: read_channel_intel(a_token[3]); break;
    case 4: read_channel_intel(a_token[4]); break;
    default: printf("Error!!!!!!\n");
  }
  printf("relu layer %d got token starting..., ir %d ic %d\n", layer_idx, *input_r, *input_c);
  #endif
  int cache_offset_ = *cache_offset;
  *cache_offset += *input_r * *input_c;
  int input_c_ = *input_c;
  int input_r_ = *input_r;
  #ifdef USING_CHANNEL
  // write_channel_intel(l_ack[layer_idx], 1);
  switch (layer_idx) {
    case 0: write_channel_intel(l_ack[0], 1); break;
    case 1: write_channel_intel(l_ack[1], 1); break;
    case 2: write_channel_intel(l_ack[2], 1); break;
    case 3: write_channel_intel(l_ack[3], 1); break;
    case 4: write_channel_intel(l_ack[4], 1); break;
    default: printf("Error!!!!!!\n");
  }
  #endif
  for (int i = 0; i < input_c_*input_r_; ++i) {
    #ifdef USING_CHANNEL
    // float nxt_num = read_channel_intel(c_ri[layer_idx]);
    float nxt_num;
    switch (layer_idx) {
      case 0: nxt_num = read_channel_intel(c_ri[0]); break;
      case 1: nxt_num = read_channel_intel(c_ri[1]); break;
      case 2: nxt_num = read_channel_intel(c_ri[2]); break;
      case 3: nxt_num = read_channel_intel(c_ri[3]); break;
      case 4: nxt_num = read_channel_intel(c_ri[4]); break;
      default: printf("Error!!!!!!\n");
    }
    cache[cache_offset_+i] = nxt_num > 0 ? 1 : 0;
    float out = nxt_num > 0 ? nxt_num : 0;
    // write_channel_intel(c_li[layer_idx+1], out);
    switch (layer_idx+1) {
      case 0: write_channel_intel(c_li[0], out); break;
      case 1: write_channel_intel(c_li[1], out); break;
      case 2: write_channel_intel(c_li[2], out); break;
      case 3: write_channel_intel(c_li[3], out); break;
      case 4: write_channel_intel(c_li[4], out); break;
      default: printf("Error!!!!!!\n");
    }
    #else
    cache[cache_offset_+i] = input_data[i] > 0 ? 1 : 0;
    input_data[i] = input_data[i] > 0 ? input_data[i] : 0;
    #endif
  }
}

__attribute__((max_global_work_dim(0)))
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
  local int c_offset;
  *cache_offset -= *input_grad_r * (*input_grad_c);
  c_offset = *cache_offset;
  for (int i = 0; i < *input_grad_r*(*input_grad_c); ++i) input_grad[i] *= cache[c_offset+i];
}

__attribute__((max_global_work_dim(0)))
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

// #ifdef USING_CHANNEL
// __attribute__((num_compute_units(5)))
// #endif
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

__attribute__((max_global_work_dim(0)))
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
    float grad_squared = pow(grads[i], 2);
    second_moment[i] = *beta2 * (second_moment[i] - grad_squared) + grad_squared;
    corrected_snd = sqrt(second_moment[i]) + *epsilon;
    corrected_fst = (first_moment[i] / corrected_snd) * lr;
    params[i] -= corrected_fst;
  }
  // printf("grad size %d\n", *grad_size);
  // for (int i = 0; i < *grad_size; ++i) {
  //   params[i] += 100;
  // }
}

__attribute__((max_global_work_dim(0)))
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

__attribute__((max_global_work_dim(0)))
__kernel void examine_int_array(__global int* restrict arr,
                                const int param_size,
                                int last_n_digit) {
  if (last_n_digit == -1) last_n_digit = param_size;
  for (int i = param_size - last_n_digit; i < param_size; ++i) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}
