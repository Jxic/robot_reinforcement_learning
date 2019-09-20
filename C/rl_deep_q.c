#include "rl_deep_q.h"
#include <stdio.h>
#include <time.h>
#include "rl_utils.h"
#include "sim_api.h"
#include "model.h"
#include "utils.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include "matrix_op.h"
#include "optimizer.h"
#include "model_utils.h"
#include "opencl_interface.h"

#define STATE_DIM 3 //16
#define ACTION_DIM 9 //4
#define APPLIED_ACTION_DIM 1
#define UNIT_MOVEMENT 2 / ACTION_DIM
#define GAMMA 0.99
#define C_LR 0.001
#define A_LR 0.001
#define EPOCH 100000
#define MAX_EPOCH_LEN 1000
#define BATCH_SIZE 128 // same as 64 timesteps
#define PRE_TRAIN_STEPS 10000
#define MEMORY_SIZE 1000000
#define RANDOM_INIT_ANGLE 1
#define RANDOM_INIT_DEST 1
#define NUM_OF_LAYERS 3
#define ACTION_BOUND 2
#define ENV_LIMIT 200
#define PROB_RAND 0.2

#define DDPG_ACTOR_FILE "DQ_ACTOR_PENDULUM.model"
// #define DDPG_ACTOR_T_FILE "DQ_ACTOR_T_PENDULUM.model"
// #define DDPG_CRITIC_FILE "DQ_CRITIC_PENDULUM.model"
// #define DDPG_CRITIC_T_FILE "DQ_CRITI_T_PENDULUM.model"

static float actions[ACTION_DIM];
static int actor_layers_config[NUM_OF_LAYERS] = {100, 100, ACTION_DIM};
static layer_type actor_layers_acts[NUM_OF_LAYERS] = {relu, relu, placeholder};
static model *actor;
static experience_buffer* exp_buf;

static int init_actor_w_target();
static int pre_training();
// static float reward(matrix_t* last, matrix_t* curr);
static matrix_t* get_action(matrix_t* state, float act_noise);
static float* run_epoch();
static float train();
static void save_all_model();
static matrix_t* deep_q_rand_action(int dim);
static matrix_t* converted_action(matrix_t* one_hot);

void run_rl_deep_q  () {
  // preparation phase

  float action_l = -2;
  for (int i = 0; i < ACTION_DIM; i++) {
    actions[i] = action_l;
    action_l += 0.5;
  }
  init_actor_w_target();
  exp_buf = init_experience_buffer(MEMORY_SIZE);

  initEnv(APPLIED_ACTION_DIM, ENV_REACHING);
  // randomly explore for certain number of steps
  printf("Initialized models\n");
  
  if (!pre_training()) {
    printf("[RUN_DDPG] failed to fill the experience buffer");
  }
  printf("Done pre-training\n");

  // OpenCL
  const char * names[] = {
    "linear_forward_prop",
    "relu_forward_prop",
    "mse",
    "relu_backward_prop",
    "transpose_params_n_cache",
    "linear_backward_prop",
    "generate_update_adam",
    "transpose_params_n_cache",
    // "matmul_engine",
    "dqn_grad",
    "transfer_data",
    #ifdef USING_CHANNEL
    "channel_start",
    "channel_end",
    "channel_manager",
    "prepare_input_grads",
    "b_channel_end",
    "b_channel_manager",
    #endif
  };
  int num_of_kernels = 10;
  #ifdef USING_CHANNEL
  num_of_kernels = 16;
  #endif
  c_init_opencl(num_of_kernels, names);
  initialize_training_env(actor, BATCH_SIZE);
  initialize_values_on_device(actor);

  // training
  int epc = 0;
  clock_t start = clock(), diff;
  while (epc < EPOCH) {
    epc++;
    float* info = run_epoch();
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Dones: %.1f | Episode: %d | Rewards: %.3f | Critic_loss: %.1f | Time elapsed: %.1f mins \n", info[1], epc, info[0], info[2], msec/(float)60000);
    free(info);
    if (epc % 50 == 0) {
      save_all_model();
    }
  }
  
  closeEnv(STATE_DIM, APPLIED_ACTION_DIM);

  free_experience_buffer(exp_buf);
  
  free_model(actor);
}

static int init_actor_w_target() {
  actor = init_model(STATE_DIM);
  for (int i = 0; i < NUM_OF_LAYERS; ++i){
    add_linear_layer(actor, actor_layers_config[i], actor_layers_acts[i]);
    matrix_t* W = actor->hidden_linears[i].data.l.W;
    if (i == NUM_OF_LAYERS-1) {
      for (int i = 0; i < W->cols*W->rows; ++i) {
        W->data[i] = rand_uniform(-0.003, 0.003);
      }
    }
  }
  init_caches(actor, BATCH_SIZE);
  compile_model(actor, no_loss, adam);
  return 1;
}

static int pre_training() {
  matrix_t* state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST, STATE_DIM, APPLIED_ACTION_DIM);
  for (int i = 0; i < PRE_TRAIN_STEPS; ++i) {
    // matrix_t* new_action = random_action(STATE_DIM, ACTION_DIM);//rand_uniform(-ACTION_BOUND, ACTION_BOUND);
    matrix_t* new_action = deep_q_rand_action(ACTION_DIM);
    matrix_t* new_action_ = converted_action(new_action);
    matrix_t* nxt_state = step(new_action_, STATE_DIM, APPLIED_ACTION_DIM);
    matrix_t* s = slice_col_wise(state, 0, STATE_DIM);
    matrix_t* s_a = concatenate(s, new_action, 1);
    matrix_t* s_a_ns = concatenate(s_a, nxt_state, 1);

    store_experience(exp_buf, s_a_ns);
    free_matrix(s_a);
    free_matrix(s);
    free_matrix(new_action);
    free_matrix(new_action_);
    if (nxt_state->data[nxt_state->cols-2]) {
      free_matrix(nxt_state);
      free_matrix(state);
      state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST, STATE_DIM, APPLIED_ACTION_DIM);
    } else {
      free_matrix(state);
      state = nxt_state;
    }
  }
  free_matrix(state);
  return 1;
}

static float* run_epoch() {
  float sum = 0;
  float dones = 0;
  float final_loss = 0;
  matrix_t* state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST, STATE_DIM, APPLIED_ACTION_DIM);
  int i;
  for (i = 0; i < MAX_EPOCH_LEN; ++i) {
    matrix_t* new_action = get_action(state, PROB_RAND);
    matrix_t* new_action_ = converted_action(new_action);
    matrix_t* nxt_state = step(new_action_, STATE_DIM, APPLIED_ACTION_DIM);
    float new_reward = nxt_state->data[nxt_state->cols-1];
    sum += new_reward;
    matrix_t* s = slice_col_wise(state, 0, STATE_DIM);
    matrix_t* s_a = concatenate(s, new_action, 1);
    matrix_t* s_a_ns = concatenate(s_a, nxt_state, 1);

    store_experience(exp_buf, s_a_ns);
    final_loss = train();
    free_matrix(s_a);
    free_matrix(s);
    free_matrix(new_action);
    free_matrix(new_action_);
    if (nxt_state->data[nxt_state->cols-2]) {
      free_matrix(nxt_state);
      break;
    } else {
      free_matrix(state);
      state = nxt_state;
    }
  }
  free_matrix(state);
  if (i < ENV_LIMIT - 1) dones += 1;
  
  float* ret = calloc(3, sizeof(float));
  ret[0] = sum;
  ret[1] = dones;
  ret[2] = final_loss;
  return ret;
}

static matrix_t* get_action(matrix_t* state, float noise_scale) {
  matrix_t* action = slice_col_wise(state, 0, STATE_DIM);
  augment_space(action, action->rows, actor->max_out);
  predict(actor, action);
  matrix_t* argmax_idx = matrix_row_argmax(action);
  matrix_t* one_hot = one_hot_encoding(argmax_idx, ACTION_DIM);

  free_matrix(argmax_idx);
  free_matrix(action);
  return one_hot;
}


static float train() {
  // training on past experiences
  matrix_t* batch = sample_experience(exp_buf, BATCH_SIZE);
  matrix_t* states = slice_col_wise(batch, 0, STATE_DIM);
  matrix_t* actions = slice_col_wise(batch, STATE_DIM, STATE_DIM+ACTION_DIM);
  matrix_t* nxt_states = slice_col_wise(batch, STATE_DIM+ACTION_DIM, 2*STATE_DIM+ACTION_DIM);
  matrix_t* dones = slice_col_wise(batch, 2*STATE_DIM+ACTION_DIM, 2*STATE_DIM+ACTION_DIM+1);
  matrix_t* rewards = slice_col_wise(batch, 2*STATE_DIM+ACTION_DIM+1, 2*STATE_DIM+ACTION_DIM+2);

  #ifndef OPENCL
  // calculating q target
  matrix_t* nxt_qs_ = matrix_clone(nxt_states);
  predict(actor, nxt_qs_);  
  
  matrix_t* nxt_max_q = new_matrix(nxt_qs_->rows, 1);
  for (int i = 0; i < nxt_qs_->rows; ++i) {
    int max = 0;
    for (int j = 0; j < nxt_qs_->cols; ++j) {
      if (nxt_qs_->data[i*nxt_qs_->cols+j] > nxt_qs_->data[i*nxt_qs_->cols+max]) max = j;
    }
    nxt_max_q->data[i] = nxt_qs_->data[i*nxt_qs_->cols+max];
  }

  mult_scalar(nxt_max_q, GAMMA);
  elem_wise_add(rewards, nxt_max_q); // reward is the target

  // get curr q gradients
  matrix_t* qs_grad = matrix_clone(states);
  predict(actor, qs_grad);

  elem_wise_mult(qs_grad, actions);
  matrix_t* qs_target = matrix_clone(actions);
  int t_idx = 0;
  for (int i = 0; i < qs_target->cols*qs_target->rows; ++i) {
    if (qs_target->data[i] > 0) {
      qs_target->data[i] = rewards->data[t_idx++];
    }
  }
  assert(t_idx == rewards->rows*rewards->cols);

  // float final_loss = mean(rewards);
  elem_wise_minus(qs_grad, qs_target);
  float final_loss = mean(qs_grad);
  mult_scalar(qs_grad, 2.0/(float)qs_target->rows);

  // backward prop
  model_backward(actor, qs_grad);
  perform_update(actor, A_LR);

  free_matrix(nxt_max_q);
  free_matrix(qs_grad);
  free_matrix(qs_target);
  free_matrix(nxt_qs_);
  free_matrix(dones);
  free_matrix(nxt_states);
  free_matrix(actions);
  free_matrix(states);
  free_matrix(batch);
  free_matrix(rewards);
  return final_loss;

  #else

  fpga_forward(actor, nxt_states, NULL);
  fpga_transfer_data_to_aux(actor);
  fpga_forward(actor, states, NULL);
  float loss = fpga_dqn_grad(actor, actions, rewards, GAMMA);
  fpga_prepare_backward(actor, BATCH_SIZE);
  fpga_backward(actor, NULL, 0);
  free_matrix(fpga_adam(actor, A_LR));

  free_matrix(dones);
  free_matrix(nxt_states);
  free_matrix(actions);
  free_matrix(states);
  free_matrix(batch);
  free_matrix(rewards);
  return loss;
  #endif


  exit(1);
  
}

static matrix_t* deep_q_rand_action(int dim) {
  matrix_t* ret = new_matrix(1, ACTION_DIM);
  int idx = rand_uniform(-1, ACTION_DIM);
  ret->data[idx] = 1;
  return ret;
}

static matrix_t* converted_action(matrix_t* one_hot) {
  matrix_t* ret = new_matrix(one_hot->rows, 1);
  matrix_t* idx = matrix_row_argmax(one_hot);
  for (int i = 0; i < ret->rows*ret->cols; ++i) {
    int nxt_idx = idx->data[i];
    assert(nxt_idx >=0 && nxt_idx <=9);
    ret->data[i] = actions[nxt_idx];
  }
  free_matrix(idx);
  return ret;
}

static void save_all_model() {
  save_model(actor, DDPG_ACTOR_FILE);
}
