#include "c_interface.hpp"
#include "setup.hpp"
#include "utils.hpp"
#include <assert.h>

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

  global_config.mem_objs.push_back(Named_buffer("params", context, CL_MEM_READ_WRITE, m->param_size*sizeof(float), NULL));
  global_config.mem_objs.push_back(Named_buffer("param_offset", context, CL_MEM_READ_WRITE, sizeof(int), NULL));
  global_config.mem_objs.push_back(Named_buffer("param_T_offset", context, CL_MEM_READ_WRITE, sizeof(int), NULL));
  global_config.mem_objs.push_back(Named_buffer("dims", context, CL_MEM_READ_WRITE, m->num_of_layers*4*sizeof(float), NULL));
  global_config.mem_objs.push_back(Named_buffer("layer_io_buffer1", context, CL_MEM_READ_WRITE, m->max_out*batch_size*sizeof(float), NULL));
  global_config.mem_objs.push_back(Named_buffer("layer_io_buffer2", context, CL_MEM_READ_WRITE, m->max_out*batch_size*sizeof(float), NULL));
  global_config.mem_objs.push_back(Named_buffer("cache", context, CL_MEM_READ_WRITE, cache_size_in_byte(m, batch_size), NULL));
  global_config.mem_objs.push_back(Named_buffer("cache_offset", context, CL_MEM_READ_WRITE, sizeof(int), NULL));
  global_config.mem_objs.push_back(Named_buffer("r1", context, CL_MEM_READ_WRITE, sizeof(int), NULL));
  global_config.mem_objs.push_back(Named_buffer("c1", context, CL_MEM_READ_WRITE, sizeof(int), NULL));
  global_config.mem_objs.push_back(Named_buffer("r2", context, CL_MEM_READ_WRITE, sizeof(int), NULL));
  global_config.mem_objs.push_back(Named_buffer("c2", context, CL_MEM_READ_WRITE, sizeof(int), NULL));
  global_config.mem_objs.push_back(Named_buffer("err_code", context, CL_MEM_READ_WRITE, sizeof(int), NULL));
  global_config.mem_objs.push_back(Named_buffer("aux_buffer", context, CL_MEM_READ_WRITE, m->max_out*batch_size*sizeof(float), NULL));
  global_config.mem_objs.push_back(Named_buffer("ret_loss", context, CL_MEM_READ_WRITE, sizeof(float), NULL));
  global_config.mem_objs.push_back(Named_buffer("fst_moment", context, CL_MEM_READ_WRITE, m->param_size*sizeof(float), NULL));
  global_config.mem_objs.push_back(Named_buffer("snd_moment", context, CL_MEM_READ_WRITE, m->param_size*sizeof(float), NULL));
  global_config.mem_objs.push_back(Named_buffer("params_T", context, CL_MEM_READ_WRITE, m->param_size*sizeof(float)+m->num_of_layers*batch_size*sizeof(float), NULL));
  global_config.mem_objs.push_back(Named_buffer("cache_T", context, CL_MEM_READ_WRITE, cache_size_in_byte(m, batch_size), NULL));
  global_config.mem_objs.push_back(Named_buffer("param_grads", context, CL_MEM_READ_WRITE, m->param_size*sizeof(float), NULL));
  global_config.mem_objs.push_back(Named_buffer("timestamp", context, CL_MEM_READ_WRITE, sizeof(int), NULL));
  global_config.mem_objs.push_back(Named_buffer("beta1", context, CL_MEM_READ_WRITE, sizeof(int), NULL));
  global_config.mem_objs.push_back(Named_buffer("beta2", context, CL_MEM_READ_WRITE, sizeof(int), NULL));
  global_config.mem_objs.push_back(Named_buffer("epsilon", context, CL_MEM_READ_WRITE, sizeof(int), NULL));
  global_config.mem_objs.push_back(Named_buffer("lr", context, CL_MEM_READ_WRITE, sizeof(int), NULL));
  global_config.mem_objs.push_back(Named_buffer("grad_size", context, CL_MEM_READ_WRITE, sizeof(int), NULL));


  size_t t_size = 0;
  for (size_t i = 0; i < global_config.mem_objs.size(); ++i) t_size += global_config.mem_objs[i].size;
  std::cout << "Total memory usage: " << t_size << std::endl;
  return 1;
}

void set_single_float_value(const char* buffer_name, float value) {
  int status;
  cl_command_queue fp_queue = global_config.command_queues[0];
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
  cl_command_queue fp_queue = global_config.command_queues[0];
  Named_buffer b = find_buffer_by_name(global_config.mem_objs, buffer_name);
  assert(b.size == sizeof(int));
  int host[1];
  *host = value;
  status = clEnqueueWriteBuffer(fp_queue, b.buffer, CL_TRUE, 0, sizeof(int), host, 0, NULL, NULL);
  std::string err_msg = std::string("Failed to set value for ") + std::string(buffer_name);
  check_status(status, err_msg.c_str());
}

void initialize_values_on_device(model* m) {
  cl_int status;
  // cl_context context = global_config.context;
  cl_command_queue fp_queue = global_config.command_queues[0];
  cl_event write_event[7];
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
  status = clEnqueueWriteBuffer(fp_queue, param_grads.buffer, CL_FALSE, 0, param_grads.size, moments_host, 0, NULL, write_event+6);
  check_status(status, "Failed to initialize value for parameter gradients");
  set_single_float_value("beta1", 0.9);
  set_single_float_value("beta2", 0.999);
  set_single_float_value("epsilon", 1e-8);
  set_single_int_value("timestamp", 0);
  clWaitForEvents(7, write_event);
  // printf("dim information\n");
  print_buffer("param_grads", 40, 1);
  // printf("params\n");
  // matrix_t* p = new_matrix(1, m->param_size);
  // p->data = params_host;
  // print_matrix(p, 1);
  // print_buffer("params", 20, 1);
  // print_buffer("fst_moment", 10, 1);
  // print_buffer("snd_moment", 10, 1);
  print_buffer("cache_T", 50, 1);
}

void enqueue_linear_forward_prop(cl_kernel linear_forward_prop,
                                 cl_command_queue fp_queue,
                                 cl_mem* params,
                                 cl_mem* param_offset,
                                 cl_mem* dims,
                                 cl_mem* input_buffer,
                                 cl_mem* input_r,
                                 cl_mem* input_c,
                                 cl_mem* cache,
                                 cl_mem* cache_offset,
                                 cl_mem* output,
                                 cl_mem* output_r,
                                 cl_mem* output_c,
                                 cl_mem* err_code,
                                 int* layer_idx,
                                 cl_event* event) {
  cl_int status;
  size_t linear_arg_idx = 0;
  status = clSetKernelArg(linear_forward_prop, linear_arg_idx++, sizeof(cl_mem), params);
  check_status(status, "[linear_forward_prop] Failed setting arg 1");
  status = clSetKernelArg(linear_forward_prop, linear_arg_idx++, sizeof(cl_mem), param_offset);
  check_status(status, "[linear_forward_prop] Failed setting arg 2");
  status = clSetKernelArg(linear_forward_prop, linear_arg_idx++, sizeof(cl_mem), dims);
  check_status(status, "[linear_forward_prop] Failed setting arg 3");
  status = clSetKernelArg(linear_forward_prop, linear_arg_idx++, sizeof(cl_mem), input_buffer);
  check_status(status, "[linear_forward_prop] Failed setting arg 4");
  status = clSetKernelArg(linear_forward_prop, linear_arg_idx++, sizeof(cl_mem), input_r);
  check_status(status, "[linear_forward_prop] Failed setting arg 5");
  status = clSetKernelArg(linear_forward_prop, linear_arg_idx++, sizeof(cl_mem), input_c);
  check_status(status, "[linear_forward_prop] Failed setting arg 6");
  status = clSetKernelArg(linear_forward_prop, linear_arg_idx++, sizeof(cl_mem), cache);
  check_status(status, "[linear_forward_prop] Failed setting arg 7");
  status = clSetKernelArg(linear_forward_prop, linear_arg_idx++, sizeof(cl_mem), cache_offset);
  check_status(status, "[linear_forward_prop] Failed setting arg 8");   
  status = clSetKernelArg(linear_forward_prop, linear_arg_idx++, sizeof(cl_mem), output);
  check_status(status, "[linear_forward_prop] Failed setting arg 9");
  status = clSetKernelArg(linear_forward_prop, linear_arg_idx++, sizeof(cl_mem), output_r);
  check_status(status, "[linear_forward_prop] Failed setting arg 10");
  status = clSetKernelArg(linear_forward_prop, linear_arg_idx++, sizeof(cl_mem), output_c);
  check_status(status, "[linear_forward_prop] Failed setting arg 11");
  status = clSetKernelArg(linear_forward_prop, linear_arg_idx++, sizeof(int), layer_idx);
  check_status(status, "[linear_forward_prop] Failed setting arg 12");
  status = clSetKernelArg(linear_forward_prop, linear_arg_idx++, sizeof(cl_mem), err_code);
  check_status(status, "[linear_forward_prop] Failed setting arg 13");
  const size_t gws = 1;
  status = clEnqueueNDRangeKernel(fp_queue, linear_forward_prop, 1, NULL, &gws, NULL, 0, NULL, event);
  check_status(status, "[linear_forward_prop] Failed enqueue linear forward propagation kernel");
}

void enqueue_relu_forward_prop(cl_kernel relu_forward_prop,
                               cl_command_queue fp_queue,
                               cl_mem* input_data,
                               cl_mem* input_r,
                               cl_mem* input_c,
                               cl_mem* cache,
                               cl_mem* cache_offset,
                               cl_mem* err_code,
                               cl_event* event) {
  cl_int status;
  size_t relu_arg_idx = 0;
  status = clSetKernelArg(relu_forward_prop, relu_arg_idx++, sizeof(cl_mem), input_data);
  check_status(status, "[relu_forward_prop] Failed setting arg 1");
  status = clSetKernelArg(relu_forward_prop, relu_arg_idx++, sizeof(cl_mem), input_r);
  check_status(status, "[relu_forward_prop] Failed setting arg 2");
  status = clSetKernelArg(relu_forward_prop, relu_arg_idx++, sizeof(cl_mem), input_c);
  check_status(status, "[relu_forward_prop] Failed setting arg 3");
  status = clSetKernelArg(relu_forward_prop, relu_arg_idx++, sizeof(cl_mem), cache);
  check_status(status, "[relu_forward_prop] Failed setting arg 4");
  status = clSetKernelArg(relu_forward_prop, relu_arg_idx++, sizeof(cl_mem), cache_offset);
  check_status(status, "[relu_forward_prop] Failed setting arg 5");
  status = clSetKernelArg(relu_forward_prop, relu_arg_idx++, sizeof(cl_mem), err_code);
  check_status(status, "[relu_forward_prop] Failed setting arg 6");
  const size_t gws = 1;
  status = clEnqueueNDRangeKernel(fp_queue, relu_forward_prop, 1, NULL, &gws, NULL, 0, NULL, event);
  check_status(status, "[relu_forward_prop] Failed enqueue relu forward propagation kernel");
}

void enqueue_mse(cl_kernel mse,
                 cl_command_queue fp_queue, 
                 cl_mem* input_data,
                 cl_mem* input_r,
                 cl_mem* input_c,
                 cl_mem* target_data,
                 cl_mem* tmp_buffer,
                 cl_mem* loss,
                 cl_mem* err_code,
                 cl_event* event) {
  cl_int status;
  size_t mse_arg_idx = 0;
  status = clSetKernelArg(mse, mse_arg_idx++, sizeof(cl_mem), input_data);
  check_status(status, "[mse] Failed setting arg 1");
  status = clSetKernelArg(mse, mse_arg_idx++, sizeof(cl_mem), input_r);
  check_status(status, "[mse] Failed setting arg 2");
  status = clSetKernelArg(mse, mse_arg_idx++, sizeof(cl_mem), input_c);
  check_status(status, "[mse] Failed setting arg 3");
  status = clSetKernelArg(mse, mse_arg_idx++, sizeof(cl_mem), target_data);
  check_status(status, "[mse] Failed setting arg 4");
  status = clSetKernelArg(mse, mse_arg_idx++, sizeof(cl_mem), tmp_buffer);
  check_status(status, "[mse] Failed setting arg 5");
  status = clSetKernelArg(mse, mse_arg_idx++, sizeof(cl_mem), loss);
  check_status(status, "[mse] Failed setting arg 6");
  status = clSetKernelArg(mse, mse_arg_idx++, sizeof(cl_mem), err_code);
  check_status(status, "[mse] Failed setting arg 7");
  const size_t gws = 1;
  status = clEnqueueNDRangeKernel(fp_queue, mse, 1, NULL, &gws, NULL, 0, NULL, event);
  check_status(status, "[mse] Failed enqueue mse kernel");
}

void enqueue_linear_backward_prop(cl_kernel linear_backward_prop,
                                  cl_command_queue bp_queue,
                                  cl_mem* params_T,
                                  cl_mem* params_T_offset,
                                  cl_mem* param_offset,
                                  cl_mem* dims,
                                  cl_mem* input_grad,
                                  cl_mem* input_grad_r,
                                  cl_mem* input_grad_c,
                                  cl_mem* cache_T,
                                  cl_mem* cache_offset,
                                  cl_mem* output_grad,
                                  cl_mem* output_grad_r,
                                  cl_mem* output_grad_c,
                                  cl_mem* param_grads,
                                  int* layer_idx,
                                  cl_mem* err_code,
                                  cl_event* event) {
  cl_int status;
  size_t linear_bp_arg_idx = 0;
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(cl_mem), params_T);
  check_status(status, "[linear_backward_prop] Failed setting arg 1");
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(cl_mem), params_T_offset);
  check_status(status, "[linear_backward_prop] Failed setting arg 2");
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(cl_mem), param_offset);
  check_status(status, "[linear_backward_prop] Failed setting arg 3");
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(cl_mem), dims);
  check_status(status, "[linear_backward_prop] Failed setting arg 4");
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(cl_mem), input_grad);
  check_status(status, "[linear_backward_prop] Failed setting arg 5");
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(cl_mem), input_grad_r);
  check_status(status, "[linear_backward_prop] Failed setting arg 6");
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(cl_mem), input_grad_c);
  check_status(status, "[linear_backward_prop] Failed setting arg 7");
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(cl_mem), cache_T);
  check_status(status, "[linear_backward_prop] Failed setting arg 8");
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(cl_mem), cache_offset);
  check_status(status, "[linear_backward_prop] Failed setting arg 9");
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(cl_mem), output_grad);
  check_status(status, "[linear_backward_prop] Failed setting arg 10");
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(cl_mem), output_grad_r);
  check_status(status, "[linear_backward_prop] Failed setting arg 11");
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(cl_mem), output_grad_c);
  check_status(status, "[linear_backward_prop] Failed setting arg 12");
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(cl_mem), param_grads);
  check_status(status, "[linear_backward_prop] Failed setting arg 13");
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(int), layer_idx);
  check_status(status, "[linear_backward_prop] Failed setting arg 14");
  status = clSetKernelArg(linear_backward_prop, linear_bp_arg_idx++, sizeof(cl_mem), err_code);
  check_status(status, "[linear_backward_prop] Failed setting arg 15");

  const size_t gws = 1;
  status = clEnqueueNDRangeKernel(bp_queue, linear_backward_prop, 1, NULL, &gws, NULL, 0, NULL, event);
  check_status(status, "[linear_backward_prop] Failed enqueue linear backward propagation kernel");
}

void enqueue_relu_backward_prop(cl_kernel relu_backward_prop,
                                cl_command_queue bp_queue,
                                cl_mem* input_grad,
                                cl_mem* input_grad_r,
                                cl_mem* input_grad_c,
                                cl_mem* cache,
                                cl_mem* cache_offset,
                                cl_mem* err_code,
                                cl_event* event) {
  cl_int status;
  size_t relu_bp_arg_idx = 0;                               
  status = clSetKernelArg(relu_backward_prop, relu_bp_arg_idx++, sizeof(cl_mem), input_grad);
  check_status(status, "[linear_backward_prop] Failed setting arg 1");
  status = clSetKernelArg(relu_backward_prop, relu_bp_arg_idx++, sizeof(cl_mem), input_grad_r);
  check_status(status, "[linear_backward_prop] Failed setting arg 2");
  status = clSetKernelArg(relu_backward_prop, relu_bp_arg_idx++, sizeof(cl_mem), input_grad_c);
  check_status(status, "[linear_backward_prop] Failed setting arg 3");
  status = clSetKernelArg(relu_backward_prop, relu_bp_arg_idx++, sizeof(cl_mem), cache);
  check_status(status, "[linear_backward_prop] Failed setting arg 4");
  status = clSetKernelArg(relu_backward_prop, relu_bp_arg_idx++, sizeof(cl_mem), cache_offset);
  check_status(status, "[linear_backward_prop] Failed setting arg 5");
  // status = clSetKernelArg(relu_backward_prop, relu_bp_arg_idx++, sizeof(cl_mem), output_grad);
  // check_status(status, "[linear_backward_prop] Failed setting arg 6");
  status = clSetKernelArg(relu_backward_prop, relu_bp_arg_idx++, sizeof(cl_mem), err_code);
  check_status(status, "[linear_backward_prop] Failed setting arg 6");

  const size_t gws = 1;
  status = clEnqueueNDRangeKernel(bp_queue, relu_backward_prop, 1, NULL, &gws, NULL, 0, NULL, event);
  check_status(status, "[relu_backward_prop] Failed enqueue relu backward propagation kernel");
}

void enqueue_generate_update_adam(cl_kernel generate_update_adam,
                                  cl_command_queue queue,
                                  cl_mem* params,
                                  cl_mem* first_moment,
                                  cl_mem* second_moment,
                                  cl_mem* grads,
                                  cl_mem* grad_size,
                                  cl_mem* time_stamp,
                                  cl_mem* beta1,
                                  cl_mem* beta2,
                                  cl_mem* epsilon,
                                  cl_mem* learning_rate,
                                  cl_mem* err_code,
                                  cl_event* event) {
  cl_int status;
  size_t adam_arg_idx = 0;
  
  status = clSetKernelArg(generate_update_adam, adam_arg_idx++, sizeof(cl_mem), params);
  check_status(status, "[generate_update_adam] Failed setting arg 1");
  status = clSetKernelArg(generate_update_adam, adam_arg_idx++, sizeof(cl_mem), first_moment);
  check_status(status, "[generate_update_adam] Failed setting arg 2");
  status = clSetKernelArg(generate_update_adam, adam_arg_idx++, sizeof(cl_mem), second_moment);
  check_status(status, "[generate_update_adam] Failed setting arg 3");
  status = clSetKernelArg(generate_update_adam, adam_arg_idx++, sizeof(cl_mem), grads);
  check_status(status, "[generate_update_adam] Failed setting arg 4");
  status = clSetKernelArg(generate_update_adam, adam_arg_idx++, sizeof(cl_mem), grad_size);
  check_status(status, "[generate_update_adam] Failed setting arg 5");
  status = clSetKernelArg(generate_update_adam, adam_arg_idx++, sizeof(cl_mem), time_stamp);
  check_status(status, "[generate_update_adam] Failed setting arg 6");
  status = clSetKernelArg(generate_update_adam, adam_arg_idx++, sizeof(cl_mem), beta1);
  check_status(status, "[generate_update_adam] Failed setting arg 7");
  status = clSetKernelArg(generate_update_adam, adam_arg_idx++, sizeof(cl_mem), beta2);
  check_status(status, "[generate_update_adam] Failed setting arg 8");
  status = clSetKernelArg(generate_update_adam, adam_arg_idx++, sizeof(cl_mem), epsilon);
  check_status(status, "[generate_update_adam] Failed setting arg 9");
  status = clSetKernelArg(generate_update_adam, adam_arg_idx++, sizeof(cl_mem), learning_rate);
  check_status(status, "[generate_update_adam] Failed setting arg 10");
  status = clSetKernelArg(generate_update_adam, adam_arg_idx++, sizeof(cl_mem), err_code);
  check_status(status, "[generate_update_adam] Failed setting arg 11");

  const size_t gws = 1;
  status = clEnqueueNDRangeKernel(queue, generate_update_adam, 1, NULL, &gws, NULL, 0, NULL, event);
  check_status(status, "[generate_update_adam] Failed enqueue generate update adam kernel");
}

void enqueue_transpose_params_n_cache(cl_kernel tpnc,
                                      cl_command_queue queue,
                                      cl_mem* params,
                                      cl_mem* cache,
                                      int* num_layers,
                                      int* batch_size,
                                      cl_mem* dims,
                                      cl_mem* params_T,
                                      cl_mem* cache_T,
                                      cl_mem* param_offset,
                                      cl_mem* err_code,
                                      cl_event* event) {
  cl_int status;
  size_t adam_arg_idx = 0;
  
  status = clSetKernelArg(tpnc, adam_arg_idx++, sizeof(cl_mem), params);
  check_status(status, "[transpose_params_n_cache] Failed setting arg 1");
  status = clSetKernelArg(tpnc, adam_arg_idx++, sizeof(cl_mem), cache);
  check_status(status, "[transpose_params_n_cache] Failed setting arg 2");
  status = clSetKernelArg(tpnc, adam_arg_idx++, sizeof(int), num_layers);
  check_status(status, "[transpose_params_n_cache] Failed setting arg 3");
  status = clSetKernelArg(tpnc, adam_arg_idx++, sizeof(int), batch_size);
  check_status(status, "[transpose_params_n_cache] Failed setting arg 4");
  status = clSetKernelArg(tpnc, adam_arg_idx++, sizeof(cl_mem), dims);
  check_status(status, "[transpose_params_n_cache] Failed setting arg 5");
  status = clSetKernelArg(tpnc, adam_arg_idx++, sizeof(cl_mem), params_T);
  check_status(status, "[transpose_params_n_cache] Failed setting arg 6");
  status = clSetKernelArg(tpnc, adam_arg_idx++, sizeof(cl_mem), cache_T);
  check_status(status, "[transpose_params_n_cache] Failed setting arg 7");
  status = clSetKernelArg(tpnc, adam_arg_idx++, sizeof(cl_mem), param_offset);
  check_status(status, "[transpose_params_n_cache] Failed setting arg 8");
  status = clSetKernelArg(tpnc, adam_arg_idx++, sizeof(cl_mem), err_code);
  check_status(status, "[transpose_params_n_cache] Failed setting arg 9");

  const size_t gws = 1;
  status = clEnqueueNDRangeKernel(queue, tpnc, 1, NULL, &gws, NULL, 0, NULL, event);
  check_status(status, "[transpose_params_n_cache] Failed enqueue transpose_params_n_cache kernel");
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

float fpga_forward(model* m, matrix_t* x, matrix_t* y) {
  // context, queue, kernels ...
  cl_int status;
  // cl_context context = global_config.context;
  cl_command_queue fp_queue = global_config.command_queues[0];
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
  cl_mem err_code = find_buffer_by_name(global_config.mem_objs, "err_code").buffer; 
  cl_mem aux_buffer = find_buffer_by_name(global_config.mem_objs, "aux_buffer").buffer;
  cl_mem loss = find_buffer_by_name(global_config.mem_objs, "ret_loss").buffer;

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
  for (int n = 0; n < m->num_of_layers-1; ++n) {
    // enqueue linear forward
    layer_idx = n;
    enqueue_linear_forward_prop(linear_forward_prop, fp_queue, &params, &param_offset, &dims, input_buffer, input_r, input_c, &cache, &cache_offset, output_buffer, output_r, output_c, &err_code, &layer_idx, &linear_fp_event);
    clWaitForEvents(1, &linear_fp_event);
    // if (n == 1) {
    //   check_buffer(m, fp_queue);
    // }
    // enqueue activation forward
    enqueue_relu_forward_prop(relu_forward_prop, fp_queue, output_buffer, output_r, output_c, &cache, &cache_offset, &err_code, &activation_fp_event);
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
    clWaitForEvents(1, &activation_fp_event);
  }
  layer_idx = m->num_of_layers - 1;
  enqueue_linear_forward_prop(linear_forward_prop, fp_queue, &params, &param_offset, &dims, input_buffer, input_r, input_c, &cache, &cache_offset, output_buffer, output_r, output_c, &err_code, &layer_idx, &linear_fp_event);
  clWaitForEvents(1, &linear_fp_event);

  // print_matrix(y, 1);

  // transfer target
  status = clEnqueueWriteBuffer(fp_queue, *input_buffer, CL_TRUE, 0, y->rows*y->cols*sizeof(float), y->data, 0, NULL, NULL);
  enqueue_mse(mse, fp_queue, output_buffer, output_r, output_c, input_buffer, &aux_buffer, &loss, &err_code, &mse_event);
  clWaitForEvents(1, &mse_event);
  // read loss
  float loss_host[1];
  status = clEnqueueReadBuffer(fp_queue, loss, CL_TRUE, 0, sizeof(float), loss_host, 1, &mse_event, NULL);



  // float params_host[m->param_size];
  // assert(m->opt.type == adam);
  // for (int i = 0; i < m->param_size; ++i) params_host[i] = *(m->opt.cache.a.trainable_params[i]);
  // matrix_t* p = new_matrix(1, m->param_size);
  // p->data = params_host;
  // p->cols = 20;
  // print_matrix(p, 1);
  // print_buffer("params", 20, 1);

  // print_buffer("layer_io_buffer1", 32*3, 1);
  // print_buffer("layer_io_buffer2", 32*3, 1);

  return loss_host[0];
}

int fpga_prepare_backward(model* m, int batch_size) {
  // context, kernels, queue ...
  cl_command_queue queue = global_config.command_queues[0];
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

  // print_buffer("params", 500 ,1);
  // print_buffer("cache", 500 ,1);
  enqueue_transpose_params_n_cache(tpnc, queue, &params, &cache, &num_layers, &batch_size, &dims, &params_T, &cache_T, &param_T_offset, &err_code, &tpnc_event);
  clWaitForEvents(1, &tpnc_event);
  
  // print_buffer("layer_io_buffer1", 500, 1);
  // print_buffer("cache_T", 500 ,1);
  // print_buffer("cache_T", 50, 1);
}

int fpga_backward(model* m, matrix_t* grad) {
  // context, kernels, queue ...
  // cl_int status;
  // cl_context context = global_config.context;
  cl_command_queue bp_queue = global_config.command_queues[0];
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
  // cl_mem cache = find_buffer_by_name(global_config.mem_objs, "cache").buffer;
  cl_mem cache_offset = find_buffer_by_name(global_config.mem_objs, "cache_offset").buffer;
  cl_mem r1 = find_buffer_by_name(global_config.mem_objs, "r1").buffer;
  cl_mem c1 = find_buffer_by_name(global_config.mem_objs, "c1").buffer;
  cl_mem r2 = find_buffer_by_name(global_config.mem_objs, "r2").buffer;
  cl_mem c2 = find_buffer_by_name(global_config.mem_objs, "c2").buffer;
  cl_mem param_grads = find_buffer_by_name(global_config.mem_objs, "param_grads").buffer;
  cl_mem err_code = find_buffer_by_name(global_config.mem_objs, "err_code").buffer; 
  cl_mem cache_T = find_buffer_by_name(global_config.mem_objs, "cache_T").buffer;
  
  // read gradients dimension
  // int g_dim_host[2];
  // status = clEnqueueReadBuffer(bp_queue, output_r, CL_TRUE, 0, sizeof(float), g_dim_host, 0, NULL, NULL);
  // status = clEnqueueReadBuffer(bp_queue, output_c, CL_TRUE, 0, sizeof(float), g_dim_host+1, 0, NULL, NULL);
  // check_status(status, "Failed reading final output dimension");

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


  int layer_idx = m->num_of_layers-1;
  cl_mem tmp;
  enqueue_linear_backward_prop(linear_backward_prop, bp_queue, &params_T, &param_T_offset, &param_offset, &dims, &input_buffer, &input_r, &input_c, &cache_T, &cache_offset, &output_buffer, &output_r, &output_c, &param_grads, &layer_idx, &err_code, &linear_bp_event);
  clWaitForEvents(1, &linear_bp_event);

  for (int i = m->num_of_layers-2; i >= 0; --i) {
    layer_idx = i;
    enqueue_relu_backward_prop(relu_backward_prop, bp_queue, &output_buffer, &output_r, &output_c, &cache_T, &cache_offset, &err_code, &activation_bp_event);
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

    enqueue_linear_backward_prop(linear_backward_prop, bp_queue, &params_T, &param_T_offset, &param_offset, &dims, &input_buffer, &input_r, &input_c, &cache_T, &cache_offset, &output_buffer, &output_r, &output_c, &param_grads, &layer_idx, &err_code, &linear_bp_event);
    clWaitForEvents(1, &linear_bp_event);
  }

  print_buffer("param_grads", 30, 1);
  float params_g_device[m->param_size];
  float params_g_host[m->param_size];
  int status;
  status = clEnqueueReadBuffer(bp_queue, param_grads, CL_TRUE, 0, sizeof(float)*m->param_size, params_g_device, 0, NULL, NULL);
  check_status(status, "Failed reading grads");
  for (int i = 0; i < m->param_size; ++i) params_g_host[i] = *(m->opt.cache.a.trainable_params_g[i]);
  matrix_t* gd = new_matrix(1, 30);
  matrix_t* gh = new_matrix(1, 30);
  gd->data = params_g_device;
  gh->data = params_g_host;
  // print_matrix(gd);
  // print_matrix(gh, 1);
  return 1;
}

matrix_t* fpga_adam(model* m, float lr) {
  assert(m->opt.type == adam);
  // matrix_t* params_host = new_matrix(1, m->param_size);
  // context, kernels, queues ...
  // cl_int status;
  // cl_context context = global_config.context;
  cl_command_queue queue = global_config.command_queues[0];
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

  // other variable
  // int grad_size = m->param_size;

  // invoke kernel
  enqueue_generate_update_adam(generate_update_adam, queue, &params, &fst_moment, &snd_moment, &param_grads, &grad_size, &timestamp, &beta1, &beta2, &epsilon, &learning_rate, &err_code, &update_event);
  clWaitForEvents(1, &update_event);

  // read out updated params
  matrix_t* updated_p = new_matrix(1, m->param_size);
  float* params_host = updated_p->data;
  int status;
  status = clEnqueueReadBuffer(queue, params, CL_TRUE, 0, sizeof(float)*m->param_size, params_host, 0, NULL, NULL);
  check_status(status, "Failed reading updated parameters");
  // for (int i = 0; i < m->param_size; ++i) *(m->opt.cache.a.trainable_params[i]) = params_host[i];
  // print_buffer("params", 40, 1);
  return updated_p;
}

int free_all_memory_objs() {

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
  cl_command_queue q = global_config.command_queues[0];
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
