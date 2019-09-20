#include "../macros.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable
#ifdef USING_CHANNEL
channel int l_token[10] __attribute__ ((depth(10)));
channel int l_ack[10] __attribute__ ((depth(10)));
channel int a_token[10] __attribute__ ((depth(10)));

channel float c_li[10] __attribute__ ((depth(10)));
channel float c_ri[10] __attribute__ ((depth(10)));
#endif

channel int linear_forward_token __attribute__ ((depth(1)));
channel int linear_backward_token __attribute__ ((depth(1)));
channel int matmul_token_forward __attribute__ ((depth(1)));
channel int matmul_token_backward __attribute__ ((depth(1)));
// mode: 0 forward, 1 backward
__kernel void matmul_engine(int mode,
            __global const float* restrict x,
            __global const int* restrict x_row,
            __global const int* restrict x_col,
            __global const int* restrict x_offset,
            __global const float* restrict y,
            __global const int* y_row,
            __global const int* y_col,
            __global const int* restrict y_offset,
            __global float* restrict r,
            __global int* restrict r_row,
            __global int* restrict r_col,
            __global const int* restrict r_offset) {

  if (!mode) {
    read_channel_intel(matmul_token_forward);
  } else {
    read_channel_intel(matmul_token_backward);
  }
  int l_r_row = *x_row;
  int l_r_col = *y_col;

  // clear output buffer
  for (int i = 0 ; i < l_r_row*l_r_col; ++i) {
    r[i+*r_offset] = 0;
  }

  for (int i = 0; i < l_r_row; ++i) {
    for (int k = 0; k < *x_col; ++k) {
      for (int j = 0; j < l_r_col; ++j) {
        r[i*l_r_col+j+(*r_offset)] += x[i*(*x_col)+k+(*x_offset)] * y[k*(*y_col)+j+(*y_offset)];
      }
    }
  }
  *r_row = l_r_row;
  *r_col = l_r_col;

  if (!mode) {
    write_channel_intel(linear_forward_token, 1);
  } else {
    write_channel_intel(linear_backward_token, 1);
  }
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
  write_channel_intel(a_token[layer_idx], 1);
  float x[1024];
  float r_[1024];
  
  for (int i = 0; i < l_r_row; ++i) {
    for (int ri = 0; ri < l_r_col; ++ri) {
      r_[ri] = 0;
    }
    for (int vi = 0; vi < x_col_; ++vi) {
      x[vi] = read_channel_intel(c_li[layer_idx]);
    }
    for (int j = 0; j < l_r_col; ++j) {
      for (int k = 0; k < x_col_; ++k) {
        r_[j] += x[k] * y[k*y_col+j+y_offset];
      }
    }

    for (int b = 0; b < l_r_col; ++b) {
      write_channel_intel(c_ri[layer_idx], r_[b] + y[bias_offset+b]);
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
  read_channel_intel(a_token[idx]);
  write_channel_intel(l_ack[idx], 1);
  for (int i = 0; i < *output_r*(*output_c); ++i) {
    buffer[offset+i] = read_channel_intel(c_ri[idx]);
  }
}

// __attribute__((max_global_work_dim(0)))
__kernel void channel_manager(const int num_of_layers) {
  for (int i = 0; i < num_of_layers; ++i) {
    printf("allowing for forward propagation of layer %d/%d\n", i+1, num_of_layers);
    write_channel_intel(l_token[i], 1);
    read_channel_intel(l_ack[i]);
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
  read_channel_intel(l_token[layer_idx]);
  printf("layer %d got token starting, input_r: %d, input_c: %d\n", layer_idx , *input_r, *input_c);
  #endif
  write_channel_intel(matmul_token_forward, 1);
  int cache_offset = *cache_offset_;
  int dim_offset = layer_idx*4;
  int W_r; W_r = dims[dim_offset];
  int W_c; W_c = dims[dim_offset+1];
  int b_r; b_r = dims[dim_offset+2];
  int b_c; b_c = dims[dim_offset+3];


  // save cache
  for (int i = 0; i < *input_r*(*input_c); ++i) cache[cache_offset+i] = input_data[i];
  *cache_offset_ += *input_r * *(input_c);

  #ifdef USING_CHANNEL
  int layer_param_offset_ = *layer_param_offset;
  *layer_param_offset += W_r * W_c + b_r * b_c;
  matmul_w_bias_from_channel(input_r, input_c, 0, params, W_r, W_c, layer_param_offset_, output, output_r, output_c, 0, layer_param_offset_ + W_r * W_c, layer_idx);
  #endif

  #ifndef USING_CHANNEL
  // matmul(input_data, *input_r, *input_c, 0, params, W_r, W_c, *layer_param_offset, output, output_r, output_c, 0);
  // *layer_param_offset += W_r * W_c;
  #endif


  
  #ifndef USING_CHANNEL
  read_channel_intel(linear_forward_token);
  *layer_param_offset += W_r*W_c;
  add_bias(output, *output_r, *output_c, params, *layer_param_offset);
  *layer_param_offset += b_r*b_c;
  #endif
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
  read_channel_intel(a_token[layer_idx]);
  printf("relu layer %d got token starting..., ir %d ic %d\n", layer_idx, *input_r, *input_c);
  #endif
  int cache_offset_ = *cache_offset;
  *cache_offset += *input_r * *input_c;
  int input_c_ = *input_c;
  int input_r_ = *input_r;
  #ifdef USING_CHANNEL
  write_channel_intel(l_ack[layer_idx], 1);
  #endif
  for (int i = 0; i < input_c_*input_r_; ++i) {
    #ifdef USING_CHANNEL
    float nxt_num = read_channel_intel(c_ri[layer_idx]);
    cache[cache_offset_+i] = nxt_num > 0 ? 1 : 0;
    float out = nxt_num > 0 ? nxt_num : 0;
    write_channel_intel(c_li[layer_idx+1], out);
    #else
    cache[cache_offset_+i] = input_data[i] > 0 ? 1 : 0;
    input_data[i] = input_data[i] > 0 ? input_data[i] : 0;
    #endif
  }
}

__attribute__((max_global_work_dim(0)))
__kernel void mse(__global const float* restrict input_data,
                  __global const int* restrict input_r,
                  __global const int* restrict input_c,
                  __global float* restrict target_data,
                  __global float* restrict tmp_buffer,
                  __global float* restrict loss,
                  __global int* restrict err_code) {
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


__kernel void transfer_data(__global const float* restrict a,
                            __global const int* restrict ir,
                            __global const int* restrict ic,
                            __global float* restrict b) {
  for (int i = 0; i < *ir*(*ic); ++i) b[i] = a[i];
}

__kernel void dqn_grad(__global const float* restrict nxt_qs,
                  __global const float* curr_qs,
                  __global float* actions,
                  __global const int* restrict input_r,
                  __global const int* restrict input_c,
                  const float gamma,
                  __global const float* reward,
                  __global float* loss) {

  int ir = *input_r;
  int ic = *input_c;
  float nxt_q_max[1000];

  for (int i = 0; i < ir; ++i) {
    int max = 0;
    for (int j = 0; j < ic; ++j) {
      if (nxt_qs[i*ic+j] > nxt_qs[i*ic+max]) max = j;
    }
    nxt_q_max[i] = gamma * nxt_qs[i*ic+max] + reward[i];
  }
  // calculate target
  float sum = 0;
  for (int i = 0; i < ir; ++i) {
    for (int j = 0; j < ic; ++j) {
      int idx = i*ic+j;
      if (actions[idx] > 0) {
        float cq = curr_qs[idx];
        float t = nxt_q_max[i];
        sum += (cq - t);
        float grad = (cq - t) * 2.0 / (float) ir;
        actions[idx] = grad;
      }
    }
  }
  
  *loss = sum / (ir*ic);
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
    for (int i = 0; i < c_c; ++i) {
      for (int j = 0; j < c_r; ++j) {
        cache_T[i*c_r+j+cache_offset] = cache[j*c_c+i+cache_offset];
      }
    }
    cache_offset += c_c*c_r;
    // cache_act
    c_c = W_c;
    for (int i = 0; i < c_c*c_r; ++i) {
      cache_T[i+cache_offset] = cache[i+cache_offset];
    }
    cache_offset += c_c*c_r;
  }

  *layer_param_offset = param_offset;
}

#ifdef USING_CHANNEL
channel int b_l_token[10] __attribute__ ((depth(10)));
channel int b_l_ack[10] __attribute__ ((depth(10)));
channel int b_a_token[10] __attribute__ ((depth(10)));
channel int preparation_signal __attribute__ ((depth(1)));;
channel float b_c_li[10] __attribute__ ((depth(10)));
channel float b_c_ri[10] __attribute__ ((depth(10)));

__kernel void b_channel_start(__global float* restrict input_buffer,
                              __global int* restrict input_r,
                              __global int* restrict input_c) {
  for (int i = 0; i < *input_r*(*input_c); ++i) {
    write_channel_intel(b_c_li[0], input_buffer[i]);
  }
}

__kernel void b_channel_end(__global float* restrict output_buffer,
                            __global int* restrict output_r,
                            __global int* restrict output_c,
                            const int idx) {
  read_channel_intel(b_a_token[idx]);
  for (int i = 0; i < *output_r*(*output_c); ++i) {
    output_buffer[i] = read_channel_intel(b_c_ri[i]);
  }
}

__kernel void b_channel_manager(const int num_of_layers) {
  read_channel_intel(preparation_signal);
  for (int i = num_of_layers - 1; i >= 0; --i) {
    write_channel_intel(b_l_token[i], 1);
    read_channel_intel(b_l_ack[i]);
  }
}

void b_matmul(__global const float* restrict x,
            const int x_row,
            const int x_col,
            const int x_offset,
            float* restrict y,
            const int y_row,
            const int y_col,
            const int y_offset,
            __global float* restrict r,
            __global int* restrict r_row,
            __global int* restrict r_col,
            const int r_offset) {
  int l_r_row; l_r_row = x_row;
  int l_r_col; l_r_col = y_col;
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

void prepare_ig_matmul(float* ig,
                       const int ig_r,
                       const int ig_c,
                       const int ig_offset,
                       __global const float* restrict pt,
                       const int pt_r,
                       const int pt_c,
                       const int pt_offset,
                       float* og,
                       int* og_r,
                       int* og_c,
                       const int og_offset) {
  int l_r_row = ig_r;
  int l_r_col = pt_c;
  for (int i = 0; i < l_r_row; ++i) {
    for (int k = 0; k < ig_c; ++k) {
      for (int j = 0; j < l_r_col; ++j) {
        og[i*l_r_col+j+og_offset] += ig[i*ig_c+k+ig_offset] * pt[k*pt_c+j+pt_offset];
      }
    }
  }

  *og_r = l_r_row;
  *og_c = l_r_col;
}

__kernel void prepare_input_grads(__global float* restrict input_grad,
                                  __global int* restrict input_r,
                                  __global int* restrict input_c,
                                  __global const float* restrict params_T,
                                  __global const int* restrict params_T_offset,
                                  __global const float* restrict cache,
                                  __global const int* restrict cache_offset,
                                  __global const int* restrict dims,
                                  const int num_of_layers) {
  int l_ir = *input_r;
  int l_ic = *input_c;
  int l_param_T_offset = *params_T_offset;
  int l_cache_offset = *cache_offset;
  int l_or = 0;
  int l_oc = 0;
  int batch_size = *input_r;
  write_channel_intel(preparation_signal, 1);
  float l_buffer_in[20000];
  float l_buffer_out[20000];
  for (int i = 0; i < 20000; i++) {
    l_buffer_in[i] = l_buffer_out[i] = 0;
  }


  for (int i = 0; i < l_ir*l_ic; ++i) {
    l_buffer_in[i] = input_grad[i];
  }

  for (int i = num_of_layers-1; i >= 0; i--) {
    int layer_offset = i*4;
    int W_r = dims[layer_offset];
    int W_c = dims[layer_offset+1];
    int b_r = dims[layer_offset+2];
    int b_c = dims[layer_offset+3];

    for (int j = 0; j < l_ir*l_ic; ++j) {
      write_channel_intel(b_c_li[i], l_buffer_in[j]);
    }
    // calculate gradient for upper layer
    l_param_T_offset -= l_ir + b_r*b_c + W_r*W_c;
    prepare_ig_matmul(l_buffer_in, l_ir, l_ic, 0, params_T, W_c, W_r, l_param_T_offset, l_buffer_out, &l_or, &l_oc, 0);
    // default to relu at the moment
    l_cache_offset -= batch_size * W_r;
    l_cache_offset -= batch_size * W_r;
    if (l_cache_offset < 0) break;
    for (int k = 0; k < batch_size*W_r; ++k) {
      l_buffer_out[k] *= cache[l_cache_offset+k];
    }

    l_ir = l_or;
    l_ic = l_oc;
    for (int m = 0; m < 20000; m++) {
      l_buffer_in[m] = l_buffer_out[m];
      l_buffer_out[m] = 0;
    }
  }

}

#endif

__kernel void relu_backward_prop(__global float* restrict input_grad,
                                 __global int* restrict input_grad_r,
                                 __global int* restrict input_grad_c,
                                 __global float* restrict cache,
                                 __global int* restrict cache_offset,
                                 int layer_idx,
                                 const int layer_idx_max,
                                 __global int* restrict err_code) {
  local int c_offset;
  *cache_offset -= *input_grad_r * (*input_grad_c);
  c_offset = *cache_offset;
  for (int i = 0; i < *input_grad_r*(*input_grad_c); ++i) input_grad[i] *= cache[c_offset+i];
}

#ifdef USING_CHANNEL
__attribute__((num_compute_units(5)))
#endif
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
                                   const int layer_idx_max,
                                   __global int* restrict err_code) {
  int layer_offset = layer_idx*4;
  int W_r; W_r = dims[layer_offset];
  int W_c; W_c = dims[layer_offset+1];
  int b_r; b_r = dims[layer_offset+2];
  int b_c; b_c = dims[layer_offset+3];
 
  for (int i = 0; i < *input_grad_r*W_r; ++i) {
    output_grad[i] = 0;
  }

  #ifdef USING_CHANNEL
  // get idx
  layer_idx = get_global_id(0);
  if (layer_idx > layer_idx_max) return;
  layer_offset = layer_idx*4;
  W_r = dims[layer_offset];
  W_c = dims[layer_offset+1];
  b_r = dims[layer_offset+2];
  b_c = dims[layer_offset+3];
  read_channel_intel(b_l_token[layer_idx]);
  // save variables and read input
  float l_buffer_in[20000];
  int l_ir = *input_grad_r;
  int l_ic = *input_grad_c;
  int l_param_T_offset = *params_T_offset;
  int l_cache_offset = *cache_offset;
  int l_layer_param_offset = *layer_param_offset;
  int l_or = 0;
  int l_oc = 0;

  *input_grad_c = W_r;
  *params_T_offset -= l_ir + b_c*b_r + W_c*W_r;
  *cache_offset -= W_r*l_ir*2;
  *layer_param_offset -= b_c*b_r + W_r*W_c;
  write_channel_intel(b_l_ack[layer_idx], 1);
  for (int i = 0; i < l_ir*l_ic; ++i) {
    l_buffer_in[i] = read_channel_intel(b_c_li[layer_idx]);
  }

  l_param_T_offset -= l_ir;
  l_layer_param_offset -= b_c*b_r;
  b_matmul(params_T, 1, l_ir, l_param_T_offset, l_buffer_in, l_ir, l_ic, 0, param_grads, output_grad_r, output_grad_c, l_layer_param_offset);

  l_param_T_offset -= b_c*b_r;
  l_layer_param_offset -= W_c*W_r;
  l_param_T_offset -= W_c*W_r;
  l_cache_offset -= W_r * l_ir;
  b_matmul(cache_T, W_r, l_ir, l_cache_offset, l_buffer_in, l_ir, l_ic, 0, param_grads, output_grad_r, output_grad_c, l_layer_param_offset);

  #else
 
  *params_T_offset -= *input_grad_r;
  *layer_param_offset -= b_c*b_r;

  write_channel_intel(matmul_token_backward, 1);
  // matmul(params_T, 1, *input_grad_r, *params_T_offset, input_grad, *input_grad_r, *input_grad_c, 0, param_grads, output_grad_r, output_grad_c, *layer_param_offset);
  read_channel_intel(linear_backward_token);
  
  *params_T_offset -= b_c*b_r;

  *layer_param_offset -= W_r * W_c;
  *params_T_offset -= W_r * W_c;
  *cache_offset -= W_r * (*input_grad_r);
  write_channel_intel(matmul_token_backward, 1);
  // matmul(cache_T, W_r, *input_grad_r, *cache_offset, input_grad, *input_grad_r, *input_grad_c, 0, param_grads, output_grad_r, output_grad_c, *layer_param_offset);
  read_channel_intel(linear_backward_token);
  write_channel_intel(matmul_token_backward, 1);

  // matmul(input_grad, *input_grad_r, *input_grad_c, 0, params_T, W_c, W_r, *params_T_offset, output_grad, output_grad_r, output_grad_c, 0);
  read_channel_intel(linear_backward_token);
 
  #endif
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

}
