#include "rl_ddpg.h"
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

#define STATE_DIM 3 //16
#define ACTION_DIM 1 //4
#define UNIT_MOVEMENT 2
#define GAMMA 0.99
#define C_LR 0.001
#define A_LR 0.0001
#define EPOCH 1000000
#define POLYAK 0.999
#define MAX_EPOCH_LEN 1000
#define BATCH_SIZE 64 // same as 64 timesteps
#define PRE_TRAIN_STEPS 10000
#define MEMORY_SIZE 1000000
#define NOISE_SCALE 0.1
#define RANDOM_INIT_ANGLE 1
#define RANDOM_INIT_DEST 1
#define NUM_OF_LAYERS 4
#define ACTION_BOUND 2
#define ENV_LIMIT 200

#define DDPG_ACTOR_FILE "DDPG_ACTOR_PENDULUM.model"
#define DDPG_ACTOR_T_FILE "DDPG_ACTOR_T_PENDULUM.model"
#define DDPG_CRITIC_FILE "DDPG_CRITIC_PENDULUM.model"
#define DDPG_CRITIC_T_FILE "DDPG_CRITI_T_PENDULUM.model"


static int actor_layers_config[NUM_OF_LAYERS] = {256, 400, 300, ACTION_DIM};
static int critic_layers_config[NUM_OF_LAYERS] = {256, 400, 300, 1};
static layer_type actor_layers_acts[NUM_OF_LAYERS] = {relu, relu, relu, tanh_};
static layer_type critic_layers_acts[NUM_OF_LAYERS] = {relu, relu, relu, placeholder};
static model *actor, *critic, *actor_target, *critic_target;
static experience_buffer* exp_buf;

static int init_actor_w_target();
static int init_critic_w_target();
static int pre_training();
// static float reward(matrix_t* last, matrix_t* curr);
static matrix_t* get_action(matrix_t* state, float act_noise);
static float* run_epoch();
static float train();
static void save_all_model();

void run_ddpg() {
  // preparation phase
  init_actor_w_target();
  init_critic_w_target();
  exp_buf = init_experience_buffer(MEMORY_SIZE);

  initEnv(ACTION_DIM, ENV_REACHING);
  // randomly explore for certain number of steps
  printf("Initialized models\n");
  
  if (!pre_training()) {
    printf("[RUN_DDPG] failed to fill the experience buffer");
  }
  printf("Done pre-training\n");

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
  
  closeEnv(STATE_DIM, ACTION_DIM);

  free_experience_buffer(exp_buf);
  
  free_model(actor);
  free_model(actor_target);
  free_model(critic);
  free_model(critic_target);
}

static int init_actor_w_target() {
  actor = init_model(STATE_DIM);
  actor_target = init_model(STATE_DIM);
  for (int i = 0; i < NUM_OF_LAYERS; ++i){
    add_linear_layer(actor, actor_layers_config[i], actor_layers_acts[i]);
    add_linear_layer(actor_target, actor_layers_config[i], actor_layers_acts[i]);
    matrix_t* W_target = actor_target->hidden_linears[i].data.l.W;
    matrix_t* b_target = actor_target->hidden_linears[i].data.l.b;
    matrix_t* W = actor->hidden_linears[i].data.l.W;
    matrix_t* b = actor->hidden_linears[i].data.l.b;
    if (i == NUM_OF_LAYERS-1) {
      for (int i = 0; i < W->cols*W->rows; ++i) {
        W->data[i] = rand_uniform(-0.003, 0.003);
      }
    }
    copy_matrix(W_target, W);
    copy_matrix(b_target, b);
  }
  init_caches(actor, BATCH_SIZE);
  init_caches(actor_target, BATCH_SIZE);
  compile_model(actor, no_loss, adam);
  compile_model(actor_target, no_loss, adam);
  return 1;
}

static int init_critic_w_target() {
  critic = init_model(STATE_DIM + ACTION_DIM);
  critic_target = init_model(STATE_DIM + ACTION_DIM);
  for (int i = 0; i < NUM_OF_LAYERS; ++i){
    add_linear_layer(critic, critic_layers_config[i], critic_layers_acts[i]);
    add_linear_layer(critic_target, critic_layers_config[i], critic_layers_acts[i]);
    matrix_t* W_target = critic_target->hidden_linears[i].data.l.W;
    matrix_t* b_target = critic_target->hidden_linears[i].data.l.b;
    matrix_t* W = critic->hidden_linears[i].data.l.W;
    matrix_t* b = critic->hidden_linears[i].data.l.b;
    if (i == NUM_OF_LAYERS-1) {
      for (int i = 0; i < W->cols*W->rows; ++i) {
        W->data[i] = rand_uniform(-0.003, 0.003);
      }
    }
    copy_matrix(W_target, W);
    copy_matrix(b_target, b);
  }
  compile_model(critic, mse_loss, adam);
  compile_model(critic_target, mse_loss, adam);
  init_caches(critic, BATCH_SIZE);
  init_caches(critic_target, BATCH_SIZE);
  return 1;
}

static int pre_training() {
  matrix_t* state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST, STATE_DIM, ACTION_DIM);
  for (int i = 0; i < PRE_TRAIN_STEPS; ++i) {
    matrix_t* new_action = random_action(STATE_DIM, ACTION_DIM);//rand_uniform(-ACTION_BOUND, ACTION_BOUND);
    matrix_t* nxt_state = step(new_action, STATE_DIM, ACTION_DIM);
    matrix_t* s = slice_col_wise(state, 0, STATE_DIM);
    matrix_t* s_a = concatenate(s, new_action, 1);
    matrix_t* s_a_ns = concatenate(s_a, nxt_state, 1);

    store_experience(exp_buf, s_a_ns);
    free_matrix(s_a);
    free_matrix(s);
    free_matrix(new_action);
    if (nxt_state->data[nxt_state->cols-2]) {
      free_matrix(nxt_state);
      free_matrix(state);
      state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST, STATE_DIM, ACTION_DIM);
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
  matrix_t* state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST, STATE_DIM, ACTION_DIM);
  int i;
  for (i = 0; i < MAX_EPOCH_LEN; ++i) {
    matrix_t* new_action = get_action(state, NOISE_SCALE);
    matrix_t* nxt_state = step(new_action, STATE_DIM, ACTION_DIM);
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
  mult_scalar(action, ACTION_BOUND);
  matrix_t* noise = rand_normal(ACTION_DIM);
  mult_scalar(noise, noise_scale);
  elem_wise_add(action, noise);
  free_matrix(noise);
  clip(action, -ACTION_BOUND, ACTION_BOUND);
  return action;
}


static float train() {
  // training on past experiences
  matrix_t* batch = sample_experience(exp_buf, BATCH_SIZE);
  matrix_t* states = slice_col_wise(batch, 0, STATE_DIM);
  matrix_t* actions = slice_col_wise(batch, STATE_DIM, STATE_DIM+ACTION_DIM);
  matrix_t* nxt_states = slice_col_wise(batch, STATE_DIM+ACTION_DIM, 2*STATE_DIM+ACTION_DIM);
  matrix_t* dones = slice_col_wise(batch, 2*STATE_DIM+ACTION_DIM, 2*STATE_DIM+ACTION_DIM+1);
  matrix_t* rewards = slice_col_wise(batch, 2*STATE_DIM+ACTION_DIM+1, 2*STATE_DIM+ACTION_DIM+2);

  // calculating critic's target
  matrix_t* nxt_actions = matrix_clone(nxt_states);
  predict(actor_target, nxt_actions);
  matrix_t* nxt_qs = concatenate(nxt_states, nxt_actions, 1);
  
  predict(critic_target, nxt_qs);
  neg(dones);
  add_scalar(dones, 1);
  elem_wise_mult(nxt_qs, dones);
  mult_scalar(nxt_qs, GAMMA);
  elem_wise_add(rewards, nxt_qs);
  // update critic
  matrix_t* qs = concatenate(states, actions, 1);
  float final_loss = fit(critic, qs, rewards, BATCH_SIZE, 1, C_LR, 0, 1);

  // find gradient of Q w.r.t action
  matrix_t* n_actions = matrix_clone(states);
  predict(actor, n_actions);
  matrix_t* q_n_actions = concatenate(states, n_actions, 1);
  predict(critic, q_n_actions);
  matrix_t* c_grad = matrix_clone(q_n_actions);
  for (int i = 0; i < c_grad->rows*c_grad->cols; ++i) {
    c_grad->data[i] = 1 / (float)c_grad->rows;
  }

  // mult_scalar(c_grad, 1/(float)c_grad->rows);
  model_backward(critic, c_grad);
  assert(c_grad->cols == STATE_DIM + ACTION_DIM);
  matrix_t* a_grad = slice_col_wise(c_grad, STATE_DIM, STATE_DIM+ACTION_DIM);
  mult_scalar(a_grad, ACTION_BOUND);
  mult_scalar(a_grad, 1/(float) a_grad->rows);
  neg(a_grad);

  // back propagation and update
  model_backward(actor, a_grad);
  perform_update(actor, A_LR);

  // update target
  int num_of_layers = actor_target->num_of_layers;
  for (int i = 0; i < num_of_layers; ++i) {
    matrix_t* W_target = actor_target->hidden_linears[i].data.l.W;
    matrix_t* b_target = actor_target->hidden_linears[i].data.l.b;
    matrix_t* W = matrix_clone(actor->hidden_linears[i].data.l.W);
    matrix_t* b = matrix_clone(actor->hidden_linears[i].data.l.b);
    mult_scalar(W, (1 - POLYAK));
    mult_scalar(b, (1 - POLYAK));
    mult_scalar(W_target, POLYAK);
    mult_scalar(b_target, POLYAK);
    elem_wise_add(W_target, W);
    elem_wise_add(b_target, b);
    free_matrix(W);
    free_matrix(b);
  }

  num_of_layers = critic_target->num_of_layers;
  for (int i = 0; i < num_of_layers; ++i) {
    matrix_t* W_target = critic_target->hidden_linears[i].data.l.W;
    matrix_t* b_target = critic_target->hidden_linears[i].data.l.b;
    matrix_t* W = matrix_clone(critic->hidden_linears[i].data.l.W);
    matrix_t* b = matrix_clone(critic->hidden_linears[i].data.l.b);
    mult_scalar(W, (1 - POLYAK));
    mult_scalar(b, (1 - POLYAK));
    mult_scalar(W_target, POLYAK);
    mult_scalar(b_target, POLYAK);
    elem_wise_add(W_target, W);
    elem_wise_add(b_target, b);
    free_matrix(W);
    free_matrix(b);
  }

  free_matrix(a_grad);
  free_matrix(c_grad);
  free_matrix(q_n_actions);
  free_matrix(n_actions);
  free_matrix(nxt_qs);
  free_matrix(dones);
  free_matrix(nxt_actions);
  free_matrix(nxt_states);
  free_matrix(actions);
  free_matrix(states);
  free_matrix(batch);
  free_matrix(qs);
  free_matrix(rewards);
  return final_loss;
}

static void save_all_model() {
  save_model(actor, DDPG_ACTOR_FILE);
  save_model(actor_target, DDPG_ACTOR_T_FILE);
  save_model(critic, DDPG_CRITIC_FILE);
  save_model(critic_target, DDPG_CRITIC_T_FILE);
}
