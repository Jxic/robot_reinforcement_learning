#include "model.h"
#include <stdlib.h>
#include <assert.h>
#include "matrix_op.h"
#include <stdio.h>
#include "layers.h"
#include "macros.h"
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "utils.h"
#include "opencl_interface.h"

static float model_forward(model* m, matrix_t* x, matrix_t* y);
static int add_activation_layer(model* m, layer_type activation);

model* init_model(int input_dim) {
  model* new_m = malloc(sizeof(model));
  new_m->input_dim = new_m->output_dim = new_m->max_out = input_dim;
  new_m->num_of_layers = 0;
  new_m->hidden_linears = (layer*)malloc(sizeof(layer));
  new_m->hidden_activations = (layer*)malloc(sizeof(layer));
  new_m->loss_layer.type = no_loss;
  new_m->cache_initialized = 0;
  new_m->opt.type = no_opt;
  new_m->param_size = 0;
  return new_m;
}

int add_linear_layer(model* m, int number_of_neurons, layer_type activation) {
  assert(number_of_neurons > 0);
  m->num_of_layers++;
  m->max_out = number_of_neurons > m->max_out ? number_of_neurons : m->max_out;
  // new linear layer
  layer linear_wrapper;
  init_linear(&linear_wrapper, m->output_dim, number_of_neurons);
  
  m->param_size += m->output_dim * number_of_neurons;
  m->param_size += number_of_neurons;
  
  // realloc m->hidden_linears to hold one more layer
  m->hidden_linears = realloc(m->hidden_linears, sizeof(layer)*m->num_of_layers);
  m->hidden_linears[m->num_of_layers-1] = linear_wrapper;

  m->output_dim = number_of_neurons;

  // add activation layer
  add_activation_layer(m, activation);

  return 1;
}

int add_conv_layer(model* m, int i_rows, int i_cols, int i_channels, int f_num, int f_size, int stride, int padding, layer_type activation) {
  // assuming squared input for now
  assert(i_rows == i_cols);
  m->num_of_layers++;

  if (((i_rows+padding*2)-f_size) % stride != 0) {
    printf("Expecting the filter size and stride to convolve every input value\n");
    exit(1);
  }
  assert(i_rows*i_cols*i_channels==m->output_dim);
  layer conv_wrapper;
  init_linear(&conv_wrapper, f_size*f_size*i_channels, f_num);
  conv_wrapper.data.l.sizes[0] = conv_wrapper.data.l.sizes[1] = f_size;
  conv_wrapper.data.l.sizes[2] = conv_wrapper.data.l.input_sizes[2] = i_channels;
  conv_wrapper.data.l.input_sizes[0] = i_rows;
  conv_wrapper.data.l.input_sizes[1] = i_cols;
  conv_wrapper.data.l.stride = stride;
  conv_wrapper.data.l.padding = padding;
  conv_wrapper.data.l.with_pooling = 0;
  conv_wrapper.type = conv;

  m->param_size += f_size*f_size*i_channels*f_num;
  m->param_size += f_num;

  m->hidden_linears = realloc(m->hidden_linears, sizeof(layer)*m->num_of_layers);
  m->hidden_linears[m->num_of_layers-1] = conv_wrapper;

  int single_filter_output_dim = (i_rows + padding * 2 - f_size) / stride + 1;
  int single_filter_output_size = pow(single_filter_output_dim, 2);
  m->output_dim = single_filter_output_size * f_num;

  conv_wrapper.data.l.t_out_size = m->output_dim;
  

  m->max_out = m->output_dim > m->max_out ? m->output_dim : m->max_out;
  printf("max %d output dim %d single_filter_out %d\n", m->max_out, m->output_dim, single_filter_output_size);
  add_activation_layer(m, activation);

  return 1;
}

int add_max_pool_layer(model* m, int i_rows, int i_cols, int i_channels, int f_size, int stride) {
  assert(m->hidden_linears[m->num_of_layers-1].type == conv);
  assert(i_rows*i_cols*i_channels == m->output_dim);
  assert(stride == f_size);

  if ((i_rows-f_size) % stride != 0) {
    printf("Expecting the filter size and stride to convolve every input value\n");
    exit(1);
  }

  int out_dim = (i_rows-f_size) / stride + 1;
  m->output_dim = out_dim*out_dim*i_channels;

  m->hidden_linears[m->num_of_layers-1].data.l.with_pooling = 1;
  layer* max_pool_wrapper = malloc(sizeof(layer));
  max_pool_layer new_max_pool;
  m->hidden_linears[m->num_of_layers-1].data.l.max_pool = max_pool_wrapper;

  m->hidden_linears[m->num_of_layers-1].data.l.t_out_size = ((i_rows-f_size)/stride+1)*i_channels;

  new_max_pool.input_sizes[0] = i_rows;
  new_max_pool.input_sizes[1] = i_cols;
  new_max_pool.input_sizes[2] = i_channels;
  new_max_pool.sizes[0] = f_size;
  new_max_pool.sizes[1] = f_size;
  new_max_pool.sizes[2] = 1;
  new_max_pool.stride = stride;
  new_max_pool.cache = new_matrix(1,1);

  max_pool_wrapper->data.max = new_max_pool;
  max_pool_wrapper->type = max_pool;

  return 1;
}

static int add_activation_layer(model* m, layer_type activation) {
    // cache size is decided at fit time
  layer activation_wrapper;
  switch (activation)
  {
    case sigmoid: {
      sigmoid_layer new_sigmoid;
      //new_sigmoid.cache = malloc(sizeof(matrix_t));
      new_sigmoid.cache = NULL;
      activation_wrapper.type = sigmoid;
      activation_wrapper.data.s = new_sigmoid;
      break;
    }
    case tanh_: {
      tanh_layer new_tanh;
      //new_tanh.cache = malloc(sizeof(matrix_t));
      new_tanh.cache = NULL;
      activation_wrapper.type = tanh_;
      activation_wrapper.data.t = new_tanh;
      break;
    }
    case relu: {
      relu_layer new_relu;
      //new_relu.cache = malloc(sizeof(matrix_t));
      new_relu.cache = NULL;
      activation_wrapper.type = relu;
      activation_wrapper.data.r = new_relu;
      break;
    }
    case placeholder: {
      placeholder_layer new_placeholder;
      new_placeholder.dummy = NULL;
      activation_wrapper.type = placeholder;
      activation_wrapper.data.p = new_placeholder;
      break;
    }
    default:
      printf("[ADD_LINEAR_LAYER] unrecognized activation\n");
      exit(1);
  }
  // realloc m->hidden_activations to hold new activation
  m->hidden_activations = realloc(m->hidden_activations, sizeof(layer)*m->num_of_layers);
  m->hidden_activations[m->num_of_layers-1] = activation_wrapper;
  return 1;
}

int compile_model(model* m, layer_type loss, optimizer_type opt_type) {
  // cache size is decided at fit time
  // initialize loss layer
  layer loss_wrapper;
  switch (loss) {
    case mse_loss: {
      mse_loss_layer new_mse_loss;
      new_mse_loss.cache_pred = NULL;
      new_mse_loss.cache_target = NULL;
      loss_wrapper.type = mse_loss;
      loss_wrapper.data.m = new_mse_loss;
      break;
    }
    case cce_loss: {
      cce_loss_layer new_cce_loss;
      new_cce_loss.cache_pred = NULL;
      new_cce_loss.cache_target = NULL;
      loss_wrapper.type = cce_loss;
      loss_wrapper.data.c = new_cce_loss;
      break;
    }
    case no_loss:
      loss_wrapper.type = no_loss;
      break;
    default:
      printf("[COMPILE_MODEL] unrecognized loss\n");
      exit(1);
  }
  m->loss_layer = loss_wrapper;

  // initialize optimizer
  optimizer opt_wrapper;
  switch (opt_type) {
    case adam: {
      init_adam(m);
      break;
    }
    case sgd: {
      sgd_optimizer new_sgd;
      new_sgd.learning_rate = SGD_LR;
      opt_wrapper.type = sgd;
      opt_wrapper.cache.s = new_sgd;
      m->opt = opt_wrapper;
      break;
    }
    case no_opt: {
      printf("[CAVEAT] Optimizer not specified\n");
      break;
    }
    default:
      break;
  }
  return 1;
}

int print_network(model* m) {
  char* names[] = {"tanh","relu", "linear", "sigmoid", "identity", "mse_loss", "no_loss", "cce_loss", "conv", "max_pool"};
  char* opt_names[] = {"sgd", "adam", "no_opt"};
  printf("---------------------------------------\n");
  printf(" input dimension: %d\n", m->input_dim);
  printf(" output dimension: %d\n", m->output_dim);
  printf(" total number of layers: %d\n", m->num_of_layers);
  printf(" max out: %d\n", m->max_out);
  printf("---------------------------------------\n");
  for (int i = 0; i < m->num_of_layers; ++i) {
    printf("layer %d:\n", i);
    printf(" %s(%d -> %d) -> %s \n", names[m->hidden_linears[i].type],
                                  m->hidden_linears[i].data.l.W->rows,
                                  m->hidden_linears[i].data.l.W->cols,
                                  names[m->hidden_activations[i].type]);
    if(m->hidden_linears[i].type == conv && m->hidden_linears[i].data.l.with_pooling) {
      printf(" max pooling (size: %d stride: %d)\n", ((layer*)m->hidden_linears[i].data.l.max_pool)->data.max.sizes[0], ((layer*)m->hidden_linears[i].data.l.max_pool)->data.max.stride);
    } 
  }
  printf("%s %s\n", names[m->loss_layer.type], opt_names[m->opt.type]);
  printf("---------------------------------------\n");
  return 1;
}



float fit(model* m, matrix_t* x, matrix_t* y, int batch_size, int epoch, float learning_rate, int shuffle, int auto_update) {
  assert(x->rows == y->rows);
  if (!m->cache_initialized && !init_caches(m, x->rows)) {
    printf("[INIT_CACHES] failed to initialize caches\n");
    exit(1);
  }
  float final_loss = 0;

  #ifdef OPENCL
  const char * names[] = {
    // "vector_add",
    // "gemm",
    "linear_forward_prop",
    "relu_forward_prop",
    "mse",
    "relu_backward_prop",
    "transpose_params_n_cache",
    "linear_backward_prop",
    "generate_update_adam",
    "examine_int_array",
    "examine_float_array",
    "transpose_params_n_cache",
    "matmul_engine",
    #ifdef USING_CHANNEL
    "channel_start",
    "channel_end",
    "channel_manager",
    "prepare_input_grads",
    "b_channel_end",
    "b_channel_manager",
    #endif
  };
  int num_of_kernels = 11;
  #ifdef USING_CHANNEL
  num_of_kernels = 16;
  #endif
  c_init_opencl(num_of_kernels, names);
  initialize_training_env(m, batch_size);
  initialize_values_on_device(m);
  #endif

  for (int epc = 0; epc < epoch; ++epc) {
    #ifdef RUN_TEST
    struct timeval t_start;

    timer_reset(&t_start);
    
    #else
    // printf("epoch %d: ", epc+1);
    #endif
    // shuffle
    if (shuffle) {
      int* idx = shuffle_row_wise(x, 0);
      shuffle_row_wise(y, idx);
      free(idx);
    }
    int data_size = x->rows;
    int start = 0;
    float loss = 0;

    struct timeval ep_t_start;
    timer_reset(&ep_t_start);
    float prep = 0;
    float forward = 0;
    float backward = 0;
    float update = 0;
    int step_count = 0;

    while (start < data_size - 1) {
      #ifdef RUN_TEST
      printf("\repoch %d: [%d]", epc+1, step_count);
      fflush(stdout);
      #endif
      step_count++;
      int curr_batch = start+batch_size<data_size ? batch_size : data_size-start;
      if (curr_batch < batch_size) {
        step_count--;
        break;
      }
      // prepare next batch
      matrix_t* next_batch = slice_row_wise(x, start, start+curr_batch);
      matrix_t* next_target = slice_row_wise(y, start, start+curr_batch);

      if (contains_nan(next_batch) || contains_nan(next_target)) {
        printf("data contains nan\n");
        print_matrix(next_batch,1);
        print_matrix(next_target,1);
        exit(1);
      }

      prep += timer_check(&ep_t_start);

      augment_space(next_batch, batch_size, m->max_out);

      // one forward and backward pass
      #ifndef OPENCL
      float step_loss_h = model_forward(m, next_batch, next_target);
      loss += step_loss_h;
      #else
      fpga_forward(m, next_batch, next_target);
      float step_loss_d = fpga_mse_loss_forward(m, next_batch, next_target);
      loss += step_loss_d;
      // printf("step loss h %f d %f\n", step_loss_h, step_loss_d);
      #endif

      forward += timer_check(&ep_t_start);
      #ifndef OPENCL
      matrix_t* grad = loss_backward(&m->loss_layer);
      model_backward(m, grad);
      #else
      fpga_prepare_backward(m, batch_size);
      fpga_backward(m, new_matrix(1,1), 0);
      #endif
      backward += timer_check(&ep_t_start);

      #ifndef OPENCL
      free_matrix(grad);
      if (auto_update) {
        perform_update(m, learning_rate);
        // matrix_t* ph = new_matrix(1, m->param_size);
        // for (int i = 0; i < m->param_size; ++i) ph->data[i] = *(m->opt.cache.a.trainable_params[i]);
        // float sum = 0;
        // float ld = 0;
        // for (int i = 0; i < m->param_size; ++i) {
        //   float difference = fabs(pd->data[i]-ph->data[i]);
        //   if (difference > ld) {
        //     ld = difference;
        //   }
        //   sum += difference;
        // }
        // printf("largest difference %e\n", ld);
        // printf("total difference on updated parameter: %e\n", sum);
      } else if (batch_size < x->rows || epoch != 1) {
        printf("[Warning] Model is not updating, entering next loop ... batch size: %d, x_size: %d, epoch: %d\n", batch_size, x->rows, epoch);
      }
      #else
      matrix_t* pd = fpga_adam(m, learning_rate);
      free_matrix(pd);
      #endif

      update += timer_check(&ep_t_start);
 
      start = start + curr_batch;
    }
    if (epc == epoch - 1) {
      final_loss = loss / (float)step_count;
    }
    #ifdef RUN_TEST
    float msec = timer_check(&t_start);
    printf("epoch %d: %f time: %.1f ms | prep: %.1f forward: %.1f backward: %.1f update %.1f\n", epc, loss / (float)step_count, msec, prep, forward, backward, update);
    fflush(stdout);
    if (loss > 1000) {
      printf("Anomalous loss %f\n", loss);
      exit(1);
    }
    #else
    //printf("%f\n", loss);
    #endif
  }
  #ifdef RUN_TEST
  printf("\n");
  #endif
  return final_loss;
}

int init_caches(model* m, int batch_size) {
  if (m->cache_initialized) {
    return 1;
  }
  m->cache_initialized = 1;
  int last_layer_out = m->input_dim;
  for (int i = 0; i < m->num_of_layers; ++i) {
    m->hidden_linears[i].data.l.cache = new_matrix(batch_size, last_layer_out);
    // printf("layer %d init with %d\n", i, batch_size*last_layer_out);
    last_layer_out = m->hidden_linears[i].data.l.W->cols;
    if (m->hidden_linears[i].type == conv) {
      last_layer_out = m->hidden_linears[i].data.l.t_out_size * batch_size;
    }
    switch (m->hidden_activations[i].type)
    {
      case sigmoid:
        m->hidden_activations[i].data.s.cache = new_matrix(batch_size, last_layer_out);
        break;
      case relu:
        m->hidden_activations[i].data.r.cache = new_matrix(batch_size, last_layer_out);
        break;
      case tanh_:
        m->hidden_activations[i].data.t.cache = new_matrix(batch_size, last_layer_out);   
      default:
        break;
    } 
  }
  switch (m->loss_layer.type) {
    case mse_loss:
      m->loss_layer.data.m.cache_pred = new_matrix(batch_size, last_layer_out);
      m->loss_layer.data.m.cache_target = new_matrix(batch_size, last_layer_out);
      break;
    case cce_loss:
      m->loss_layer.data.c.cache_pred = new_matrix(batch_size, last_layer_out);
      m->loss_layer.data.c.cache_target = new_matrix(batch_size, last_layer_out);
    default:
      break;
  }
  return 1;
}


int predict(model* m, matrix_t* x) {
  assert(x->rows > 0 && x->cols > 0);
  augment_space(x, x->rows, m->max_out);
  if (!m->cache_initialized) {
    printf("[CAVEAT] Automatically initializing cache with size %d\n", x->rows);
    init_caches(m, x->rows);
  }
  for (int i = 0; i < m->num_of_layers; ++i) {
    // if (i == 0) {
    //   print_matrix(x, 0);
    // }
    if (!forward(m->hidden_linears+i, x)) {
      printf("[MODEL_FORWARD] failed at %dth linear layer\n", i);
      return 0;
    }

    if (!forward(m->hidden_activations+i, x)) {
      printf("[MODEL_FORWARD] failed at %dth activation layer\n", i);
      return 0;
    }
  }
  return 1;
}

static float model_forward(model* m, matrix_t* x, matrix_t* y) {
  if (!predict(m, x)) {
    exit(1);
  }

  float loss = loss_forward(&m->loss_layer, x, y);
  return loss;
}

int model_backward(model* m, matrix_t* grad) {
  augment_space(grad, grad->rows, m->max_out);
  for (int i = m->num_of_layers-1; i >= 0; --i) {
    if (!backward(m->hidden_activations+i, grad)) {
      printf("[MODEL_BACKWARD] failed at %dth activation layer\n", i);
      exit(1);
    }
    if (!backward(m->hidden_linears+i, grad)) {
      printf("[MODEL_BACKWARD] failed at %dth linear layer\n", i);
      exit(1);
    }
  }
  return 1;
}


float eval(model* m, matrix_t* x, matrix_t* y, matrix_t* min_max) {
  float sum = 0;
  matrix_t* min_max_y = slice_col_wise(min_max, x->cols, min_max->cols);
  augment_space(x, x->rows, m->max_out);
  predict(m, x);
  scale(x, min_max_y);
  scale(y, min_max_y);
  sum = loss_forward(&m->loss_layer, x, y);
  return sum;
}

int free_model(model* m) {
  for (int i = 0; i < m->num_of_layers; ++i) {
    free_layer(m->hidden_activations[i]);
    free_layer(m->hidden_linears[i]);
  }
  free_layer(m->loss_layer);
  free(m->hidden_linears);
  free(m->hidden_activations);
  free_optimizer(m->opt);
  free(m);
  return 1;
}
