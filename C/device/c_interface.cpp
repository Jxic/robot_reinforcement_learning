#include "c_interface.hpp"
#include "setup.hpp"
#include "utils.hpp"
#include <assert.h>
#include "../macros.h"
// #include <cstdarg>
#include <stdarg.h>

extern Config global_config;

void set_single_float_value(const char* buffer_name, float value);
void set_single_int_value(const char* buffer_name, int value);

int cache_size_in_byte(model* m, int batch_size) {
  int cache_size = 0;
  int in = m->input_dim;
  for (int i = 0; i < m->num_of_layers; ++i) {
    cache_size += in * batch_size;
    in = m->hidden_linears[i].data.l.W->cols;
    cache_size += batch_size * in;
  }
  return cache_size * sizeof(float);
};
int initialize_training_env(model* m, int batch_size) {
  /*
  FOR FORWARD PROP
  1. allocate memory buffer for all params
  2. allocate memory for layer params offset and set to 0
  3. allocate memory buffer for dimension information
  4. allocate memory buffer for the input batch
    (large enough to hold largest intermediate matrix)
    (used interchangeably with 7)
  5. allocate memory buffer for cache
  6. allocate memory buffer for cache offset and set to 0
  7. allocate memory buffer for output 
    (large enough to hold largest intermediate matrix)
    (used interchangeably with 4)
  8. allocate memory for output row value and column value
    (could be unintialized)
  9. allocate memory for error code

  FOR MSE
  1. allocate memory buffer for target data (use 1.4)
  2. allocate auxiliary buffer same size as target buffer
  3. allocate memory for returned loss

  FOR BACKWARD PROP
  1. allocate memory buffer for output gradients (use 1.7)

  FOR ADAM UPDATE
  1. allocate 2 more buffers for first and second moment
  2. initialize first second and second moment
  */
  
  // cl_int status;
  cl_context context = global_config.context;

  global_config.mem_objs.emplace_back("params", context, CL_MEM_READ_WRITE, m->param_size*sizeof(float), (void*)NULL);
  global_config.mem_objs.emplace_back("param_offset", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("param_T_offset", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("dims", context, CL_MEM_READ_WRITE, m->num_of_layers*4*sizeof(float), (void*)NULL);
  global_config.mem_objs.emplace_back("layer_io_buffer1", context, CL_MEM_READ_WRITE, m->max_out*batch_size*sizeof(float), (void*)NULL);
  global_config.mem_objs.emplace_back("layer_io_buffer2", context, CL_MEM_READ_WRITE, m->max_out*batch_size*sizeof(float), (void*)NULL);
  global_config.mem_objs.emplace_back("cache", context, CL_MEM_READ_WRITE, cache_size_in_byte(m, batch_size), (void*)NULL);
  global_config.mem_objs.emplace_back("cache_offset", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("r1", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("c1", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("r2", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("c2", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("err_code", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("aux_buffer", context, CL_MEM_READ_WRITE, m->max_out*batch_size*sizeof(float), (void*)NULL);
  global_config.mem_objs.emplace_back("ret_loss", context, CL_MEM_READ_WRITE, sizeof(float), (void*)NULL);
  global_config.mem_objs.emplace_back("fst_moment", context, CL_MEM_READ_WRITE, m->param_size*sizeof(float), (void*)NULL);
  global_config.mem_objs.emplace_back("snd_moment", context, CL_MEM_READ_WRITE, m->param_size*sizeof(float), (void*)NULL);
  global_config.mem_objs.emplace_back("params_T", context, CL_MEM_READ_WRITE, m->param_size*sizeof(float)+m->num_of_layers*batch_size*sizeof(float), (void*)NULL);
  global_config.mem_objs.emplace_back("cache_T", context, CL_MEM_READ_WRITE, cache_size_in_byte(m, batch_size), (void*)NULL);
  global_config.mem_objs.emplace_back("param_grads", context, CL_MEM_READ_WRITE, m->param_size*sizeof(float), (void*)NULL);
  global_config.mem_objs.emplace_back("timestamp", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("beta1", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("beta2", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("offset_ph_0", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("offset_ph_1", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("w_r", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("w_c", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("epsilon", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("lr", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);
  global_config.mem_objs.emplace_back("grad_size", context, CL_MEM_READ_WRITE, sizeof(int), (void*)NULL);

  size_t t_size = 0;
  for (size_t i = 0; i < global_config.mem_objs.size(); ++i) t_size += global_config.mem_objs[i].size;
  std::cout << "Total memory usage: " << t_size << std::endl;
  return 1;
}

void set_single_float_value(const char* buffer_name, float value) {
  int status;
  cl_command_queue fp_queue = find_queue_by_name(global_config.command_queues, "dq").q;
  Named_buffer b = find_buffer_by_name(global_config.mem_objs, buffer_name);
  assert(b.size == sizeof(float));
  float host[1];
  *host = value;
  status = clEnqueueWriteBuffer(fp_queue, b.buffer, CL_TRUE, 0, sizeof(float), host, 0, NULL, NULL);
  std::string err_msg = std::string("Failed to set value for ") + std::string(buffer_name);
  check_status(status, err_msg.c_str());
}

void set_single_int_value(const char* buffer_name, int value) {
  int status;
  cl_command_queue fp_queue = find_queue_by_name(global_config.command_queues, "dq").q;
  Named_buffer b = find_buffer_by_name(global_config.mem_objs, buffer_name);
  assert(b.size == sizeof(int));
  int host[1];
  *host = value;
  status = clEnqueueWriteBuffer(fp_queue, b.buffer, CL_TRUE, 0, sizeof(int), host, 0, NULL, NULL);
  std::string err_msg = std::string("Failed to set value for ") + std::string(buffer_name);
  check_status(status, err_msg.c_str());
}

void initialize_values_on_device(model* m) {
  printf("Initializing buffer values on device\n");
  cl_int status;
  // cl_context context = global_config.context;
  cl_command_queue fp_queue = find_queue_by_name(global_config.command_queues, "dq").q;
  cl_event write_event[6];
  // transfer params
  Named_buffer params = find_buffer_by_name(global_config.mem_objs, "params");
  float params_host[m->param_size];
  assert(m->opt.type == adam);
  for (int i = 0; i < m->param_size; ++i) params_host[i] = *(m->opt.cache.a.trainable_params[i]);
  status = clEnqueueWriteBuffer(fp_queue, params.buffer, CL_FALSE, 0, params.size, params_host, 0, NULL, write_event);
  check_status(status, "Failed transferring parameters to device");
  // set dimension info
  int dims_host[m->num_of_layers*4];
  for (int i = 0; i < 2*m->num_of_layers; i+=2) {
    int w_rows = m->hidden_linears[i/2].data.l.W->rows;
    int w_cols = m->hidden_linears[i/2].data.l.W->cols;
    int b_rows = m->hidden_linears[i/2].data.l.b->rows;
    int b_cols = m->hidden_linears[i/2].data.l.b->cols;
    int dims_offset = (i/2)*4;
    // printf("writing to %d wr %d wc %d br %d bc %d\n", dims_offset, w_rows, w_cols, b_rows, b_cols);
    dims_host[dims_offset] = w_rows;
    dims_host[dims_offset+1] = w_cols;
    dims_host[dims_offset+2] = b_rows;
    dims_host[dims_offset+3] = b_cols;
  }
  Named_buffer dims = find_buffer_by_name(global_config.mem_objs, "dims");
  status = clEnqueueWriteBuffer(fp_queue, dims.buffer, CL_FALSE, 0, dims.size, dims_host, 0, NULL, write_event+1);
  check_status(status, "Failed set dimension information");
  // set other values
  int offset_host[1];
  *offset_host = 0;
  status = clEnqueueWriteBuffer(fp_queue, find_buffer_by_name(global_config.mem_objs, "param_offset").buffer, CL_FALSE, 0, sizeof(int), offset_host, 0, NULL, write_event+2);
  check_status(status, "Failed to set parameter offset");
  status = clEnqueueWriteBuffer(fp_queue, find_buffer_by_name(global_config.mem_objs, "cache_offset").buffer, CL_FALSE, 0, sizeof(int), offset_host, 0, NULL, write_event+3);
  check_status(status, "Failed to set cache offset");
  float moments_host[m->param_size];
  for (int i = 0; i < m->param_size; ++i) moments_host[i] = 0;
  Named_buffer fst_moment = find_buffer_by_name(global_config.mem_objs, "fst_moment");
  Named_buffer snd_moment = find_buffer_by_name(global_config.mem_objs, "snd_moment");
  Named_buffer param_grads = find_buffer_by_name(global_config.mem_objs, "param_grads");
  status = clEnqueueWriteBuffer(fp_queue, fst_moment.buffer, CL_FALSE, 0, fst_moment.size, moments_host, 0, NULL, write_event+4);
  check_status(status, "Failed to initialize value for first moment");
  status = clEnqueueWriteBuffer(fp_queue, snd_moment.buffer, CL_FALSE, 0, snd_moment.size, moments_host, 0, NULL, write_event+5);
  check_status(status, "Failed to initialize value for second moment");
  set_single_float_value("beta1", 0.9);
  set_single_float_value("beta2", 0.999);
  set_single_float_value("epsilon", 1e-8);
  set_single_int_value("timestamp", 0);
  set_single_int_value("offset_ph_0", 0);
  set_single_int_value("offset_ph_1", 1);
  set_single_int_value("grad_size", m->param_size);

  // create enough command queues
  global_config.command_queues.emplace_back("mm", global_config.context, global_config.device_id, CL_QUEUE_PROFILING_ENABLE);
  global_config.command_queues.emplace_back("mm2", global_config.context, global_config.device_id, CL_QUEUE_PROFILING_ENABLE);
  global_config.command_queues.emplace_back("mm3", global_config.context, global_config.device_id, CL_QUEUE_PROFILING_ENABLE);

  #ifdef USING_CHANNEL
  global_config.command_queues.emplace_back("linear_0", global_config.context, global_config.device_id, CL_QUEUE_PROFILING_ENABLE);
  global_config.command_queues.emplace_back("relu_0", global_config.context, global_config.device_id, CL_QUEUE_PROFILING_ENABLE);
  global_config.command_queues.emplace_back("cm", global_config.context, global_config.device_id, CL_QUEUE_PROFILING_ENABLE);
  global_config.command_queues.emplace_back("b_cm", global_config.context, global_config.device_id, CL_QUEUE_PROFILING_ENABLE);
  global_config.command_queues.emplace_back("b_ce", global_config.context, global_config.device_id, CL_QUEUE_PROFILING_ENABLE);

  global_config.command_queues.emplace_back("ce", global_config.context, global_config.device_id, CL_QUEUE_PROFILING_ENABLE);
  global_config.command_queues.emplace_back("cs", global_config.context, global_config.device_id, CL_QUEUE_PROFILING_ENABLE);
  #endif
  clWaitForEvents(6, write_event);
}


void enqueue_NDRangeKernel(const char* kernel_name, cl_command_queue queue, size_t gws, cl_event* event, const char* fmt...) {
  // printf("launching %s\n", kernel_name);
  va_list args;
  va_start(args, fmt);

  cl_kernel kernel = find_kernel_by_name(global_config.kernels, kernel_name).k;
  size_t idx = 0;
  cl_int status;
  while (*fmt != '\0') {
    if (*fmt == 'm') {
      // printf("setting memory object at %ld\n", idx);
      cl_mem* m = va_arg(args, cl_mem*);
      status = clSetKernelArg(kernel, idx++, sizeof(cl_mem), m);
    } else if (*fmt == 'd') {
      // printf("setting integer object at %ld\n", idx);
      int* i = va_arg(args, int*);
      status = clSetKernelArg(kernel, idx++, sizeof(int), i);
    } else if (*fmt == 'f') {
      float* f = va_arg(args, float*);
      status = clSetKernelArg(kernel, idx++, sizeof(float), f);
    } else {
      printf("Could not recognize the format %c, aborting ...\n", *fmt);
      exit(1);
    }
    check_status(status, "[%s] Failed to set argument %d", kernel_name, idx-1);
    ++fmt;
  }
  va_end(args);
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gws, NULL, 0, NULL, event);
  check_status(status, "[%s] Failed enqueue %s kernel", kernel_name, kernel_name);
}

void check_buffer(model* m, cl_command_queue fp_queue) {
  Named_buffer b1 = find_buffer_by_name(global_config.mem_objs, "layer_io_buffer1");
  Named_buffer b2 = find_buffer_by_name(global_config.mem_objs, "layer_io_buffer2");
  matrix_t* o = new_matrix(32, 100);
  cl_int status = clEnqueueReadBuffer(fp_queue, b2.buffer, CL_TRUE, 0, 32*100*sizeof(float), o->data, 0, NULL, NULL);
  matrix_t* w = new_matrix(100, 100);
  matrix_t* b = new_matrix(1, 100);
  matrix_t* wx = new_matrix(32, 100);
  float params_host_W[100*100];
  float params_host_b[1*100];
  for (int i = 400; i < 100*100+400; ++i) params_host_W[i-400] = *(m->opt.cache.a.trainable_params[i]);
  for (int i = 400+10000; i < 100+400+10000; ++i) params_host_b[i-10400] = *(m->opt.cache.a.trainable_params[i]);
  w->data = params_host_W;
  b->data = params_host_b;
  matmul(o, w, wx);
  wx->rows = 1;
  wx->cols = 30;
  w->rows = 1;
  w->cols = 30;
  o->rows = 1;
  o->cols = 30;
  printf("result calculated using device data\n");
  printf("W\n");
  print_matrix(w,1);
  printf("In\n");
  print_matrix(o, 1);
  printf("result\n");
  print_matrix(wx,1);
}

int fpga_forward(model* m, matrix_t* x, matrix_t* y) {
  // context, queue, kernels ...
  cl_int status;
  cl_command_queue fp_queue = find_queue_by_name(global_config.command_queues, "dq").q;
  cl_kernel linear_forward_prop = find_kernel_by_name(global_config.kernels, "linear_forward_prop").k;
  cl_kernel relu_forward_prop = find_kernel_by_name(global_config.kernels, "relu_forward_prop").k;
  cl_kernel mse = find_kernel_by_name(global_config.kernels, "mse").k;
  
  // events
  cl_event linear_fp_event;
  cl_event activation_fp_event;
  cl_event mse_event;

  // find buffers
  cl_mem params = find_buffer_by_name(global_config.mem_objs, "params").buffer;
  cl_mem param_offset = find_buffer_by_name(global_config.mem_objs, "param_offset").buffer;
  cl_mem dims = find_buffer_by_name(global_config.mem_objs, "dims").buffer;
  cl_mem layer_io_buffer1 = find_buffer_by_name(global_config.mem_objs, "layer_io_buffer1").buffer;
  cl_mem layer_io_buffer2 = find_buffer_by_name(global_config.mem_objs, "layer_io_buffer2").buffer;
  cl_mem cache = find_buffer_by_name(global_config.mem_objs, "cache").buffer;
  cl_mem cache_offset = find_buffer_by_name(global_config.mem_objs, "cache_offset").buffer;
  cl_mem r1 = find_buffer_by_name(global_config.mem_objs, "r1").buffer;
  cl_mem c1 = find_buffer_by_name(global_config.mem_objs, "c1").buffer;
  cl_mem r2 = find_buffer_by_name(global_config.mem_objs, "r2").buffer;
  cl_mem c2 = find_buffer_by_name(global_config.mem_objs, "c2").buffer;
  cl_mem offset_ph_0 = find_buffer_by_name(global_config.mem_objs, "offset_ph_0").buffer;
  cl_mem w_r = find_buffer_by_name(global_config.mem_objs, "w_r").buffer;
  cl_mem w_c = find_buffer_by_name(global_config.mem_objs, "w_c").buffer;
  cl_mem err_code = find_buffer_by_name(global_config.mem_objs, "err_code").buffer; 
  cl_mem aux_buffer = find_buffer_by_name(global_config.mem_objs, "aux_buffer").buffer;
  cl_mem loss = find_buffer_by_name(global_config.mem_objs, "ret_loss").buffer;
  Named_buffer param_grads = find_buffer_by_name(global_config.mem_objs, "param_grads");

  // clean up buffers
  float moments_host[m->param_size];
  for (int i = 0; i < m->param_size; ++i) moments_host[i] = 0;
  status = clEnqueueWriteBuffer(fp_queue, param_grads.buffer, CL_TRUE, 0, param_grads.size, moments_host, 0, NULL, NULL);
  check_status(status, "Failed to initialize value for parameter gradients");
  set_single_int_value("param_offset", 0);
  set_single_int_value("cache_offset", 0);

  // transfer input batch and target batch
  status = clEnqueueWriteBuffer(fp_queue, layer_io_buffer1, CL_TRUE, 0, x->rows*x->cols*sizeof(float), x->data, 0, NULL, NULL);
  check_status(status, "Failed transferring input data");

  // set input r and c
  set_single_int_value("r1", x->rows);
  set_single_int_value("c1", x->cols);

  // prop
  cl_mem* input_buffer = &layer_io_buffer1;
  cl_mem* output_buffer = &layer_io_buffer2;
  cl_mem* input_r = &r1;
  cl_mem* input_c = &c1;
  cl_mem* output_r = &r2;
  cl_mem* output_c = &c2;
  int layer_idx;
  cl_mem* tmp;
  int mode = 0;
  // other variable
  int dims_host[m->num_of_layers*4];
  for (int i = 0; i < 2*m->num_of_layers; i+=2) {
    int w_rows = m->hidden_linears[i/2].data.l.W->rows;
    int w_cols = m->hidden_linears[i/2].data.l.W->cols;
    int b_rows = m->hidden_linears[i/2].data.l.b->rows;
    int b_cols = m->hidden_linears[i/2].data.l.b->cols;
    int dims_offset = (i/2)*4;
    // printf("writing to %d wr %d wc %d br %d bc %d\n", dims_offset, w_rows, w_cols, b_rows, b_cols);
    dims_host[dims_offset] = w_rows;
    dims_host[dims_offset+1] = w_cols;
    dims_host[dims_offset+2] = b_rows;
    dims_host[dims_offset+3] = b_cols;
  }
  printf("starting forward\n");
  for (int i = 0; i < 12; ++i) {
    printf("%d ", dims_host[i]);
  }
  printf("\n");
  #ifdef USING_CHANNEL
  // start channel
  enqueue_NDRangeKernel("channel_start", find_queue_by_name(global_config.command_queues, "cs").q, 1, NULL, "mmm", input_buffer, input_r, input_c);
  enqueue_NDRangeKernel("channel_manager", find_queue_by_name(global_config.command_queues, "cm").q, 1, NULL, "d", &m->num_of_layers);
  string layer_name;
  #endif
  int layer_idx_max = m->num_of_layers-1;
  #ifndef USING_CHANNEL
  for (int n = 0; n < m->num_of_layers-1; ++n) {
    // printf("layer %d\n", n);
    // enqueue linear forward
    layer_idx = n;
    printf("host dim %d %d\n", dims_host[layer_idx*4], dims_host[layer_idx*4+1]);
    set_single_int_value("w_r", dims_host[layer_idx*4]);
    set_single_int_value("w_c", dims_host[layer_idx*4+1]);
    printf("enqueueing linear\n");
    enqueue_NDRangeKernel("linear_forward_prop", fp_queue, 1, &linear_fp_event, "mmmmmmmmmmmddm", &params, &param_offset, &dims, input_buffer, input_r, input_c, &cache, &cache_offset, output_buffer, output_r, output_c, &layer_idx, &layer_idx_max,&err_code);
    printf("enqueueing matmul\n");
    enqueue_NDRangeKernel("matmul_engine", find_queue_by_name(global_config.command_queues, "mm").q, 1, NULL, "dmmmmmmmmmmmm", &mode, input_buffer, input_r, input_c, &offset_ph_0, &params, &w_r, &w_c, &param_offset, output_buffer, output_r, output_c, &offset_ph_0);
    clWaitForEvents(1, &linear_fp_event);
    
    // enqueue activation forward
    enqueue_NDRangeKernel("relu_forward_prop", fp_queue, 1, &activation_fp_event, "mmmmmddm", output_buffer, output_r, output_c, &cache, &cache_offset, &layer_idx, &layer_idx_max, &err_code);
    clWaitForEvents(1, &activation_fp_event);

    // swap input output buffer
    tmp = input_buffer;
    input_buffer = output_buffer;
    output_buffer = tmp;
    // swap input output dims
    tmp = input_r;
    input_r = output_r;
    output_r = tmp;
    tmp = input_c;
    input_c = output_c;
    output_c = tmp;
  }
  layer_idx = m->num_of_layers - 1;
  set_single_int_value("w_r", dims_host[layer_idx*4]);
  set_single_int_value("w_c", dims_host[layer_idx*4+1]);
  enqueue_NDRangeKernel("linear_forward_prop", fp_queue, 1, &linear_fp_event, "mmmmmmmmmmmddm", &params, &param_offset, &dims, input_buffer, input_r, input_c, &cache, &cache_offset, output_buffer, output_r, output_c, &layer_idx, &layer_idx_max, &err_code);
  enqueue_NDRangeKernel("matmul_engine", find_queue_by_name(global_config.command_queues, "mm").q, 1, NULL, "dmmmmmmmmmmmm", &mode, input_buffer, input_r, input_c, &offset_ph_0, &params, &w_r, &w_c, &param_offset, output_buffer, output_r, output_c, &offset_ph_0);

  clWaitForEvents(1, &linear_fp_event);
  #else
  enqueue_NDRangeKernel("linear_forward_prop", find_queue_by_name(global_config.command_queues, "linear_0").q, (size_t) m->num_of_layers, NULL, "mmmmmmmmmmmddm", &params, &param_offset, &dims, input_buffer, input_r, input_c, &cache, &cache_offset, output_buffer, output_r, output_c, &layer_idx, &layer_idx_max, &err_code);

  int relu_layer_idx_max = layer_idx_max - 1;
  enqueue_NDRangeKernel("relu_forward_prop", find_queue_by_name(global_config.command_queues, "relu_0").q, (size_t) layer_idx_max, NULL, "mmmmmddm", output_buffer, output_r, output_c, &cache, &cache_offset, &layer_idx, &relu_layer_idx_max, &err_code);  
  
  // end channel
  int output_offset = 0;
  layer_idx = m->num_of_layers - 1;
  cl_event end_event;
  enqueue_NDRangeKernel("channel_end", find_queue_by_name(global_config.command_queues, "ce").q, 1, &end_event, "mmmdd", output_buffer, output_r, output_c, &output_offset, &layer_idx);
  clWaitForEvents(1, &end_event);
  // clWaitForEvents(1, &linear_fp_event);
  #endif

  // // transfer target
  // status = clEnqueueWriteBuffer(fp_queue, *input_buffer, CL_TRUE, 0, y->rows*y->cols*sizeof(float), y->data, 0, NULL, NULL);
  // enqueue_NDRangeKernel("mse", fp_queue, 1, &mse_event, "mmmmmmm", output_buffer, output_r, output_c, input_buffer, &aux_buffer, &loss, &err_code);
  // clWaitForEvents(1, &mse_event);

  // // read loss
  // float loss_host[1];
  // status = clEnqueueReadBuffer(fp_queue, loss, CL_TRUE, 0, sizeof(float), loss_host, 1, &mse_event, NULL);

  // return loss_host[0];
  return 1;
}

float fpga_mse_loss_forward(model* m, matrix_t* x, matrix_t* y) {
  int status;
  // context, kernels, queue ...
  cl_command_queue fp_queue = find_queue_by_name(global_config.command_queues, "dq").q;
  
  // events
  cl_event mse_event;

  // find buffers
  cl_mem layer_io_buffer1 = find_buffer_by_name(global_config.mem_objs, "layer_io_buffer1").buffer;
  cl_mem layer_io_buffer2 = find_buffer_by_name(global_config.mem_objs, "layer_io_buffer2").buffer;
  cl_mem r1 = find_buffer_by_name(global_config.mem_objs, "r1").buffer;
  cl_mem c1 = find_buffer_by_name(global_config.mem_objs, "c1").buffer;
  cl_mem r2 = find_buffer_by_name(global_config.mem_objs, "r2").buffer;
  cl_mem c2 = find_buffer_by_name(global_config.mem_objs, "c2").buffer;
  cl_mem err_code = find_buffer_by_name(global_config.mem_objs, "err_code").buffer; 
  cl_mem loss = find_buffer_by_name(global_config.mem_objs, "ret_loss").buffer;
  cl_mem aux_buffer = find_buffer_by_name(global_config.mem_objs, "aux_buffer").buffer;

  cl_mem input_buffer;
  cl_mem output_buffer;
  cl_mem input_r;
  cl_mem input_c;
  cl_mem output_r;
  cl_mem output_c;

  if (m->num_of_layers%2==0) {
    // printf("mse using 2\n");
    input_buffer = layer_io_buffer2;
    output_buffer = layer_io_buffer1;
    input_r = r2;
    input_c = c2;
    output_r = r1;
    output_c = c1;
  } else {
    // printf("mse using 1\n");
    input_buffer = layer_io_buffer1;
    output_buffer = layer_io_buffer2;
    input_r = r1;
    input_c = c1;
    output_r = r2;
    output_c = c2;
  }
  // transfer target
  status = clEnqueueWriteBuffer(fp_queue, input_buffer, CL_TRUE, 0, y->rows*y->cols*sizeof(float), y->data, 0, NULL, NULL);
  enqueue_NDRangeKernel("mse", fp_queue, 1, &mse_event, "mmmmmmm", &output_buffer, &output_r, &output_c, &input_buffer, &aux_buffer, &loss, &err_code);
  clWaitForEvents(1, &mse_event);

  // read loss
  float loss_host[1];
  status = clEnqueueReadBuffer(fp_queue, loss, CL_TRUE, 0, sizeof(float), loss_host, 1, &mse_event, NULL);
  return loss_host[0];
}

int fpga_prepare_backward(model* m, int batch_size) {
  // context, kernels, queue ...
  cl_command_queue queue = find_queue_by_name(global_config.command_queues, "dq").q;
  cl_kernel tpnc = find_kernel_by_name(global_config.kernels, "transpose_params_n_cache").k;

  // event
  cl_event tpnc_event;

  //find buffers
  cl_mem params = find_buffer_by_name(global_config.mem_objs, "params").buffer;
  cl_mem param_T_offset = find_buffer_by_name(global_config.mem_objs, "param_T_offset").buffer;
  cl_mem cache = find_buffer_by_name(global_config.mem_objs, "cache").buffer;
  cl_mem dims = find_buffer_by_name(global_config.mem_objs, "dims").buffer;
  cl_mem params_T = find_buffer_by_name(global_config.mem_objs, "params_T").buffer;
  cl_mem cache_T = find_buffer_by_name(global_config.mem_objs, "cache_T").buffer;
  cl_mem err_code = find_buffer_by_name(global_config.mem_objs, "err_code").buffer;

  // other variable
  int num_layers = m->num_of_layers;
  int b_s = batch_size;

  enqueue_NDRangeKernel("transpose_params_n_cache", queue, 1, &tpnc_event, "mmddmmmmm", &params, &cache, &num_layers, &batch_size, &dims, &params_T, &cache_T, &param_T_offset, &err_code);
  clWaitForEvents(1, &tpnc_event);
}

matrix_t* retrieve_grad_of_input(model* m, int batch_size) {
  cl_command_queue queue = find_queue_by_name(global_config.command_queues, "dq").q;

  cl_event retrieve_event;
  // find buffers
  Named_buffer layer_io_buffer1 = find_buffer_by_name(global_config.mem_objs, "layer_io_buffer1");
  Named_buffer layer_io_buffer2 = find_buffer_by_name(global_config.mem_objs, "layer_io_buffer2");
  cl_mem r1 = find_buffer_by_name(global_config.mem_objs, "r1").buffer;
  cl_mem c1 = find_buffer_by_name(global_config.mem_objs, "c1").buffer;
  cl_mem r2 = find_buffer_by_name(global_config.mem_objs, "r2").buffer;
  cl_mem c2 = find_buffer_by_name(global_config.mem_objs, "c2").buffer;
  
  // other variable
  // cl_mem input_buffer;
  // cl_mem output_buffer;
  // cl_mem input_r;
  // cl_mem input_c;
  // cl_mem output_r;
  // cl_mem output_c;

  // reset offset
  // if (m->num_of_layers%2==0) {
  //   input_buffer = layer_io_buffer2;
  //   output_buffer = layer_io_buffer1;
  //   input_r = r2;
  //   input_c = c2;
  //   output_r = r1;
  //   output_c = c1;
  // } else {
  //   input_buffer = layer_io_buffer1;
  //   output_buffer = layer_io_buffer2;
  //   input_r = r1;
  //   input_c = c1;
  //   output_r = r2;
  //   output_c = c2;
  // }

  matrix_t* grad2 = new_matrix(batch_size, m->input_dim);
  cl_int status = clEnqueueReadBuffer(queue, layer_io_buffer2.buffer, CL_TRUE, 0, batch_size*m->input_dim*sizeof(float), grad2->data, 0, NULL, NULL);
  grad2->rows = 1;
  grad2->cols = 30;
  printf("device last grad2\n");
  print_matrix(grad2, 1);
  return grad2;
}

int fpga_backward(model* m, matrix_t* grad, int use_new_grad) {
  // context, kernels, queue ...
  cl_command_queue bp_queue = find_queue_by_name(global_config.command_queues, "dq").q;
  cl_kernel linear_backward_prop = find_kernel_by_name(global_config.kernels, "linear_backward_prop").k;
  cl_kernel relu_backward_prop = find_kernel_by_name(global_config.kernels, "relu_backward_prop").k;
  
  // events
  cl_event linear_bp_event;
  cl_event activation_bp_event;

  // find buffers
  cl_mem params_T = find_buffer_by_name(global_config.mem_objs, "params_T").buffer;
  cl_mem param_offset = find_buffer_by_name(global_config.mem_objs, "param_offset").buffer;
  cl_mem param_T_offset = find_buffer_by_name(global_config.mem_objs, "param_T_offset").buffer;
  cl_mem dims = find_buffer_by_name(global_config.mem_objs, "dims").buffer;
  cl_mem layer_io_buffer1 = find_buffer_by_name(global_config.mem_objs, "layer_io_buffer1").buffer;
  cl_mem layer_io_buffer2 = find_buffer_by_name(global_config.mem_objs, "layer_io_buffer2").buffer;
  cl_mem cache_offset = find_buffer_by_name(global_config.mem_objs, "cache_offset").buffer;
  cl_mem r1 = find_buffer_by_name(global_config.mem_objs, "r1").buffer;
  cl_mem c1 = find_buffer_by_name(global_config.mem_objs, "c1").buffer;
  cl_mem r2 = find_buffer_by_name(global_config.mem_objs, "r2").buffer;
  cl_mem c2 = find_buffer_by_name(global_config.mem_objs, "c2").buffer;
  cl_mem w_r = find_buffer_by_name(global_config.mem_objs, "w_r").buffer;
  cl_mem w_c = find_buffer_by_name(global_config.mem_objs, "w_c").buffer;
  cl_mem offset_ph_0 = find_buffer_by_name(global_config.mem_objs, "offset_ph_0").buffer;
  cl_mem offset_ph_1 = find_buffer_by_name(global_config.mem_objs, "offset_ph_1").buffer;
  cl_mem param_grads = find_buffer_by_name(global_config.mem_objs, "param_grads").buffer;
  cl_mem err_code = find_buffer_by_name(global_config.mem_objs, "err_code").buffer; 
  cl_mem cache_T = find_buffer_by_name(global_config.mem_objs, "cache_T").buffer;
  
  // other variable
  cl_mem input_buffer;
  cl_mem output_buffer;
  cl_mem input_r;
  cl_mem input_c;
  cl_mem output_r;
  cl_mem output_c;

  // reset offset
  if (m->num_of_layers%2==0) {
    input_buffer = layer_io_buffer2;
    output_buffer = layer_io_buffer1;
    input_r = r1;
    input_c = c1;
    output_r = r2;
    output_c = c2;
  } else {
    input_buffer = layer_io_buffer1;
    output_buffer = layer_io_buffer2;
    input_r = r2;
    input_c = c2;
    output_r = r1;
    output_c = c1;
  }

  // set in new grad if needed
  if (use_new_grad) {
    set_single_int_value("r1", grad->rows);
    set_single_int_value("c1", grad->cols);
    set_single_int_value("r2", grad->rows);
    set_single_int_value("c2", grad->cols);
    clEnqueueWriteBuffer(bp_queue, input_buffer, CL_TRUE, 0, grad->rows*grad->cols*sizeof(float), grad->data, 0, NULL, NULL);
  }

  int dims_host[m->num_of_layers*4];
  for (int i = 0; i < 2*m->num_of_layers; i+=2) {
    int w_rows = m->hidden_linears[i/2].data.l.W->rows;
    int w_cols = m->hidden_linears[i/2].data.l.W->cols;
    int b_rows = m->hidden_linears[i/2].data.l.b->rows;
    int b_cols = m->hidden_linears[i/2].data.l.b->cols;
    int dims_offset = (i/2)*4;
    // printf("writing to %d wr %d wc %d br %d bc %d\n", dims_offset, w_rows, w_cols, b_rows, b_cols);
    dims_host[dims_offset] = w_rows;
    dims_host[dims_offset+1] = w_cols;
    dims_host[dims_offset+2] = b_rows;
    dims_host[dims_offset+3] = b_cols;
  }

  int mode = 1;
  int layer_idx = m->num_of_layers-1;
  #ifndef USING_CHANNEL
  cl_mem tmp;
  int layer_idx_max = m->num_of_layers-1;
  set_single_int_value("w_r", dims_host[layer_idx*4]);
  set_single_int_value("w_c", dims_host[layer_idx*4+1]);
  enqueue_NDRangeKernel("linear_backward_prop", bp_queue, 1, &linear_bp_event, "mmmmmmmmmmmmmddm", &params_T, &param_T_offset, &param_offset, &dims, &input_buffer, &input_r, &input_c, &cache_T, &cache_offset, &output_buffer, &output_r, &output_c, &param_grads, &layer_idx, &layer_idx_max, &err_code);
  enqueue_NDRangeKernel("matmul_engine", find_queue_by_name(global_config.command_queues, "mm").q, 1, NULL, "dmmmmmmmmmmmm", &mode, &params_T, &offset_ph_1, &input_r, &param_T_offset, &input_buffer, &input_r, &input_c, &offset_ph_0, &param_grads, &output_r, &output_c, &param_offset);
  enqueue_NDRangeKernel("matmul_engine", find_queue_by_name(global_config.command_queues, "mm2").q, 1, NULL, "dmmmmmmmmmmmm", &mode, &cache_T, &w_r, &input_r, &cache_offset, &input_buffer, &input_r, &input_c, &offset_ph_0, &param_grads, &output_r, &output_c, &param_offset);
  enqueue_NDRangeKernel("matmul_engine", find_queue_by_name(global_config.command_queues, "mm3").q, 1, NULL, "dmmmmmmmmmmmm", &mode, &input_buffer, &input_r, &input_c, &offset_ph_0, &params_T, &w_c, &w_r, &param_T_offset, &output_buffer, &output_r, &output_c, &offset_ph_0);
  clWaitForEvents(1, &linear_bp_event);

  for (int i = m->num_of_layers-2; i >= 0; --i) {
    layer_idx = i;
    enqueue_NDRangeKernel("relu_backward_prop", bp_queue, 1, &activation_bp_event, "mmmmmddm", &output_buffer, &output_r, &output_c, &cache_T, &cache_offset, &layer_idx, &layer_idx_max, &err_code);

    // swap input ouput buffer
    tmp = input_buffer;
    input_buffer = output_buffer;
    output_buffer = tmp;
    // swap input out dims
    tmp = input_r;
    input_r = output_r;
    output_r = tmp;
    tmp = input_c;
    input_c = output_c;
    output_c = tmp;
    
    clWaitForEvents(1, &activation_bp_event);
    set_single_int_value("w_r", dims_host[layer_idx*4]);
    set_single_int_value("w_c", dims_host[layer_idx*4+1]);
    enqueue_NDRangeKernel("linear_backward_prop", bp_queue, 1, &linear_bp_event, "mmmmmmmmmmmmmddm", &params_T, &param_T_offset, &param_offset, &dims, &input_buffer, &input_r, &input_c, &cache_T, &cache_offset, &output_buffer, &output_r, &output_c, &param_grads, &layer_idx, &layer_idx_max, &err_code);
    enqueue_NDRangeKernel("matmul_engine", find_queue_by_name(global_config.command_queues, "mm").q, 1, NULL, "dmmmmmmmmmmmm", &mode, &params_T, &offset_ph_1, &input_r, &param_T_offset, &input_buffer, &input_r, &input_c, &offset_ph_0, &param_grads, &output_r, &output_c, &param_offset);
    enqueue_NDRangeKernel("matmul_engine", find_queue_by_name(global_config.command_queues, "mm2").q, 1, NULL, "dmmmmmmmmmmmm", &mode, &cache_T, &w_r, &input_r, &cache_offset, &input_buffer, &input_r, &input_c, &offset_ph_0, &param_grads, &output_r, &output_c, &param_offset);
    enqueue_NDRangeKernel("matmul_engine", find_queue_by_name(global_config.command_queues, "mm3").q, 1, NULL, "dmmmmmmmmmmmm", &mode, &input_buffer, &input_r, &input_c, &offset_ph_0, &params_T, &w_c, &w_r, &param_T_offset, &output_buffer, &output_r, &output_c, &offset_ph_0);
 
    clWaitForEvents(1, &linear_bp_event);
  }

  #else
  enqueue_NDRangeKernel("b_channel_manager", find_queue_by_name(global_config.command_queues, "b_cm").q, 1, NULL, "d", &m->num_of_layers);

  enqueue_NDRangeKernel("prepare_input_grads", bp_queue, 1, NULL, "mmmmmmmmd", &input_buffer, &input_r, &input_c, &params_T, &param_T_offset, &cache_T, &cache_offset, &dims, &m->num_of_layers);

  cl_event end_event;
  enqueue_NDRangeKernel("linear_backward_prop", find_queue_by_name(global_config.command_queues, "linear_0").q, m->num_of_layers, &end_event, "mmmmmmmmmmmmmddm", &params_T, &param_T_offset, &param_offset, &dims, &input_buffer, &input_r, &input_c, &cache_T, &cache_offset, &output_buffer, &output_r, &output_c, &param_grads, &layer_idx, &m->num_of_layers-1, &err_code);

  // int end_idx = 0;

  // enqueue_NDRangeKernel("b_channel_end", find_queue_by_name(global_config.command_queues, "b_ce").q, 1, &end_event, "mmmd", &output_buffer, &output_r, &output_c, &end_idx);

  clWaitForEvents(1, &end_event);

  #endif

  // float params_g_device[m->param_size];
  // float params_g_host[m->param_size];
  // int status;
  // status = clEnqueueReadBuffer(bp_queue, param_grads, CL_TRUE, 0, sizeof(float)*m->param_size, params_g_device, 0, NULL, NULL);
  // check_status(status, "Failed reading grads");
  // for (int i = 0; i < m->param_size; ++i) params_g_host[i] = *(m->opt.cache.a.trainable_params_g[i]);
  // matrix_t* gd = new_matrix(1, 30);
  // matrix_t* gh = new_matrix(1, 30);
  // gd->data = params_g_device;
  // printf("device grads\n");
  // print_matrix(gd, 1);
  // gh->data = params_g_host;
  return 1;
}

matrix_t* fpga_adam(model* m, float lr) {
  assert(m->opt.type == adam);
  // context, kernels, queues ...

  cl_command_queue queue = find_queue_by_name(global_config.command_queues, "dq").q;
  cl_kernel generate_update_adam = find_kernel_by_name(global_config.kernels, "generate_update_adam").k;
  set_single_float_value("lr", lr);
  // events
  cl_event update_event;

  // find buffers
  cl_mem params = find_buffer_by_name(global_config.mem_objs, "params").buffer;
  cl_mem fst_moment = find_buffer_by_name(global_config.mem_objs, "fst_moment").buffer;
  cl_mem snd_moment = find_buffer_by_name(global_config.mem_objs, "snd_moment").buffer;
  cl_mem param_grads = find_buffer_by_name(global_config.mem_objs, "param_grads").buffer;
  cl_mem err_code = find_buffer_by_name(global_config.mem_objs, "err_code").buffer; 
  cl_mem timestamp = find_buffer_by_name(global_config.mem_objs, "timestamp").buffer; 
  cl_mem beta1 = find_buffer_by_name(global_config.mem_objs, "beta1").buffer; 
  cl_mem beta2 = find_buffer_by_name(global_config.mem_objs, "beta2").buffer; 
  cl_mem epsilon = find_buffer_by_name(global_config.mem_objs, "epsilon").buffer; 
  cl_mem learning_rate = find_buffer_by_name(global_config.mem_objs, "lr").buffer;
  cl_mem grad_size = find_buffer_by_name(global_config.mem_objs, "grad_size").buffer;


  // invoke kernel
  enqueue_NDRangeKernel("generate_update_adam", queue, 1, &update_event, "mmmmmmmmmmm", &params, &fst_moment, &snd_moment, &param_grads, &grad_size, &timestamp, &beta1, &beta2, &epsilon, &learning_rate, &err_code);
  clWaitForEvents(1, &update_event);

  // read out updated params
  matrix_t* updated_p = new_matrix(1, m->param_size);
  float* params_host = updated_p->data;
  int status;
  status = clEnqueueReadBuffer(queue, params, CL_TRUE, 0, sizeof(float)*m->param_size, params_host, 0, NULL, NULL);
  check_status(status, "Failed reading updated parameters");
  for (int i = 0; i < m->param_size; ++i) *(m->opt.cache.a.trainable_params[i]) = params_host[i];

  return updated_p;
}

int free_all_memory_objs() {
  global_config.mem_objs.clear();
  return 1;
}

void enqueue_examine_array(cl_kernel k, cl_command_queue q, cl_mem* b, int* s, int* ln, cl_event* event) {
  cl_int status;
  size_t arg_idx = 0;
  
  status = clSetKernelArg(k, arg_idx++, sizeof(cl_mem), b);
  check_status(status, "[examine_float_array] Failed setting arg 1");
  status = clSetKernelArg(k, arg_idx++, sizeof(int), s);
  check_status(status, "[examine_float_array] Failed setting arg 1");
  status = clSetKernelArg(k, arg_idx++, sizeof(int), ln);
  check_status(status, "[examine_float_array] Failed setting arg 1");
  
  const size_t gws = 1;
  status = clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, NULL, 0, NULL, event);
  check_status(status, "[examine_float_array] Failed enqueue examine float array kernel");
}

void print_buffer(const char* buffer_name, int last_n_digit, int type) {
  // type 0 int, type 1 float
  Named_buffer b = find_buffer_by_name(global_config.mem_objs ,buffer_name);
  cl_event event;
  printf("-------------------------------------\n");
  cl_command_queue q = find_queue_by_name(global_config.command_queues, "dq").q;
  Named_kernel examine_array;
  if (type) {
    examine_array = find_kernel_by_name(global_config.kernels, "examine_float_array");
  } else {
    examine_array = find_kernel_by_name(global_config.kernels, "examine_int_array");
  }
  int size = b.size/4;
  int ln = last_n_digit;
  enqueue_examine_array(examine_array.k, q, &b.buffer, &size, &ln, &event);
  clWaitForEvents(1, &event);
  printf("-------------------------------------\n");
  printf("Name: %s\n", buffer_name);
  printf("Vector Size: %d\n", (int)b.size/4);
  printf("Printing last %d digit(s)\n", last_n_digit);
  printf("-------------------------------------\n");
  
}
