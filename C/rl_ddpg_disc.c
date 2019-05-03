#include "rl_ddpg_disc.h"
#include <stdio.h>
#include <time.h>
#include "rl_utils.h"
#include "sim_api.h"
#include "model.h"
#include "utils.h"
#include <assert.h>
#include <math.h>

#define STATE_DIM 9
#define ACTION_DIM 3
#define UNIT_MOVEMENT 2
#define GAMMA 0.99
#define LR 0.001
#define EPOCH 1000
#define POLYAK 0.995
#define MAX_EPOCH_LEN 2000
#define BATCH_SIZE 100 // same as 200 timesteps
#define PRE_TRAIN_STEPS 10000
#define MEMORY_SIZE 1000000
#define NOISE_SCALE 0.1
#define RANDOM_INIT_ANGLE 1
#define RANDOM_INIT_DEST 1
#define NUM_OF_LAYERS 3
#define ACTION_BOUND 3.14


static int actor_layers_config[NUM_OF_LAYERS] = {100, 100, ACTION_DIM};
static int critic_layers_config[NUM_OF_LAYERS] = {100, 100, 1};
static layer_type actor_layers_acts[NUM_OF_LAYERS] = {relu, relu, tanh_};
static layer_type critic_layers_acts[NUM_OF_LAYERS] = {relu, relu, placeholder};
static model *actor, *critic, *actor_target, *critic_target;
static experience_buffer* exp_buf;

static int init_actor_w_target();
static int init_critic_w_target();
static int pre_training();
//static void test_agent();
static double reward(matrix_t* last, matrix_t* curr);
static matrix_t* get_action(matrix_t* state, double act_noise);
static double run_epoch();
static void train();

void run_ddpg() {
  // preparation phase
  if (!init_actor_w_target() ||
      !init_critic_w_target() ||
      !(exp_buf = init_experience_buffer(MEMORY_SIZE)) ||
      !initEnv()) {
    printf("[RUN_DDPG] failed to initialze the algorithm");
  }
  // randomly explore for certain number of steps
  if (!pre_training()) {
    printf("[RUN_DDPG] failed to fill the experience buffer");
  }
  // training
  int epc = 0;
  clock_t start = clock(), diff;
  while (epc < EPOCH) {
    epc++;
    double reward = run_epoch();
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Episode: %d | Rewards: %.1f | Time elapsed: %.1f mins \n", epc, reward, msec/(double)60000);
    train();
  }
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
    copy_matrix(W_target, W);
    copy_matrix(b_target, b);
  }
  init_caches(actor, BATCH_SIZE);
  init_caches(actor_target, BATCH_SIZE);
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
    copy_matrix(W_target, W);
    copy_matrix(b_target, b);
  }
  compile_model(critic, mse_loss);
  compile_model(critic_target, mse_loss);
  init_caches(critic, BATCH_SIZE);
  init_caches(critic_target, BATCH_SIZE);
  return 1;
}

static int pre_training() {
  matrix_t* state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST);
  for (int i = 0; i < PRE_TRAIN_STEPS; ++i) {
    matrix_t* new_action = new_matrix(1, ACTION_DIM);
    for (int i = 0; i < ACTION_DIM; ++i) new_action->data[i] = rand_uniform(-ACTION_BOUND, ACTION_BOUND);
    matrix_t* nxt_state = step(new_action);
    double new_reward = reward(state, nxt_state);
    matrix_t* s = slice_col_wise(state, 0, STATE_DIM);
    matrix_t* s_a = concatenate(s, new_action, 1);
    matrix_t* s_a_ns = concatenate(s_a, nxt_state, 1);
    augment_space(s_a_ns, 1, s_a_ns->cols+1);
    s_a_ns->data[s_a_ns->cols] = new_reward;
    s_a_ns->cols++;
    store_experience(exp_buf, s_a_ns);
    free_matrix(s_a);
    free_matrix(s);
    free_matrix(new_action);
    if (nxt_state->data[nxt_state->cols-1]) {
      free_matrix(nxt_state);
      free_matrix(state);
      state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST);
    } else {
      free_matrix(state);
      state = nxt_state;
    }
  }
  return 1;
}

static double run_epoch() {
  double sum = 0;
  matrix_t* state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST);
  for (int i = 0; i < MAX_EPOCH_LEN; ++i) {
    matrix_t* new_action = get_action(state, 0);
    matrix_t* nxt_state = step(new_action);
    double new_reward = reward(state, nxt_state);
    matrix_t* s = slice_col_wise(state, 0, STATE_DIM);
    matrix_t* s_a = concatenate(s, new_action, 1);
    matrix_t* s_a_ns = concatenate(s_a, nxt_state, 1);
    augment_space(s_a_ns, 1, s_a_ns->cols+1);
    s_a_ns->data[s_a_ns->cols] = new_reward;
    s_a_ns->cols++;
    store_experience(exp_buf, s_a_ns);
    free_matrix(s_a);
    free_matrix(s);
    free_matrix(new_action);
    if (nxt_state->data[nxt_state->cols-1]) {
      free_matrix(nxt_state);
      free_matrix(state);
      state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST);
    } else {
      free_matrix(state);
      state = nxt_state;
    }
  }
  return sum;
}

static matrix_t* get_action(matrix_t* state, double act_noise) {
  matrix_t* action = slice_col_wise(state, 0, STATE_DIM);
  augment_space(action, action->rows, actor->max_out);
  predict(actor, action);
  matrix_t* noise = rand_normal(ACTION_DIM);
  mult_scalar(noise, act_noise);
  elem_wise_add(action, noise);
  clip(action, -ACTION_BOUND, ACTION_BOUND);
  return action;
}

static double euclidean_distance(matrix_t* a, matrix_t* b) {
  assert(a->rows == 1 && b->rows == 1);
  assert(a->cols == b->cols);
  double sum = 0;
  for (int i = 0; i < a->cols; ++i) {
    sum += pow(a->data[i]-b->data[i], 2);
  }
  return sqrt(sum);
}

static double reward(matrix_t* last, matrix_t* curr) {
  matrix_t* l_end = slice_col_wise(last, 3, 6);
  matrix_t* l_dst = slice_col_wise(last, 6, 9);
  matrix_t* c_end = slice_col_wise(curr, 3, 6);
  matrix_t* c_dst = slice_col_wise(curr, 6, 9);
  double l_dist = euclidean_distance(l_end, l_dst);
  double c_dist = euclidean_distance(c_end, c_dst);
  if (l_dist < c_dist) {
    return -1;
  } else {
    return 1;
  }
}


static void train() {
  // training on past experiences
  matrix_t* batch = sample_experience(exp_buf, BATCH_SIZE);
  matrix_t* states = slice_col_wise(batch, 0, STATE_DIM);
  matrix_t* actions = slice_col_wise(batch, STATE_DIM, STATE_DIM+ACTION_DIM);
  matrix_t* nxt_states = slice_col_wise(batch, STATE_DIM+ACTION_DIM, 2*STATE_DIM+ACTION_DIM);
  matrix_t* dones = slice_col_wise(batch, 2*STATE_DIM+ACTION_DIM, 2*STATE_DIM+ACTION_DIM+1);
  matrix_t* rewards = slice_col_wise(batch, 2*STATE_DIM+ACTION_DIM+1, 2*STATE_DIM+ACTION_DIM+2);

  // calculating critic's target
  matrix_t* nxt_actions = clone(nxt_states);
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
  fit(critic, qs, rewards, BATCH_SIZE, 1, LR, 0);

  // find gradient of Q w.r.t action
  matrix_t* n_actions = clone(states);
  predict(actor, n_actions);
  matrix_t* q_n_actions = concatenate(states, n_actions, 1);
  predict(critic, q_n_actions);
  matrix_t* c_grad = clone(q_n_actions);
  for (int i = 0; i < c_grad->rows*c_grad->cols; ++i) {
    c_grad->data[i] = c_grad->data[i] < 0 ? -1 : 0;
  }
  mult_scalar(c_grad, 1/(double)c_grad->rows);
  model_backward(critic, c_grad);
  matrix_t* a_grad = slice_col_wise(c_grad, STATE_DIM, STATE_DIM+ACTION_DIM);
  // back propagation and update
  model_backward(actor, a_grad);
  model_update(actor, LR);

  // update target
  int num_of_layers = actor_target->num_of_layers;
  for (int i = 0; i < num_of_layers; ++i) {
    matrix_t* W_target = actor_target->hidden_linears[i].data.l.W;
    matrix_t* b_target = actor_target->hidden_linears[i].data.l.b;
    matrix_t* W = clone(actor->hidden_linears[i].data.l.W);
    matrix_t* b = clone(actor->hidden_linears[i].data.l.b);
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
}
