#include "rl_ddpg_pixel.h"
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
#include <string.h>

#define STACK_NUM 4
#define BASE_STATE_DIM 83*83*1
#define STACKED_BASE_STATE_DIM BASE_STATE_DIM*STACK_NUM
#define AUX_STATE_DIM 3
#define STATE_DIM STACKED_BASE_STATE_DIM+AUX_STATE_DIM //16
#define RECV_STATE_DIM BASE_STATE_DIM+AUX_STATE_DIM
#define ACTION_DIM 1 //4
#define UNIT_MOVEMENT 2
#define GAMMA 0.99
#define C_LR 0.001
#define A_LR 0.0001
#define EPOCH 1000000
#define POLYAK 0.999
#define MAX_EPOCH_LEN 1000
#define BATCH_SIZE 64 // same as 64 timesteps
#define PRE_TRAIN_STEPS 2000
#define MEMORY_SIZE 1000000
#define NOISE_SCALE 0.1
#define RANDOM_INIT_ANGLE 1
#define RANDOM_INIT_DEST 1
#define ACTION_BOUND 2
#define ENV_LIMIT 200


#define DDPG_ACTOR_FILE "DDPG_ACTOR_PENDULUM.model"
#define DDPG_ACTOR_T_FILE "DDPG_ACTOR_T_PENDULUM.model"
#define DDPG_CRITIC_FILE "DDPG_CRITIC_PENDULUM.model"
#define DDPG_CRITIC_T_FILE "DDPG_CRITI_T_PENDULUM.model"

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
static int modified_predict(model* m, matrix_t* input);

matrix_t** stacked_frame;
matrix_t* last_full_state;

matrix_t* stack_new_frame(matrix_t* new_frame, int is_new_episode) {
  // printf("new frame row %d col %d\n", new_frame->rows, new_frame->cols);
  assert(new_frame->cols == BASE_STATE_DIM+AUX_STATE_DIM+2);
  matrix_t* base_state = slice_col_wise(new_frame, 0, BASE_STATE_DIM);
  matrix_t* aux_state = slice_col_wise(new_frame, BASE_STATE_DIM, new_frame->cols);
  free_matrix(new_frame);
  new_frame = base_state;
  free_matrix(last_full_state);
  last_full_state = aux_state;
  matrix_t* ret;

  ret = new_matrix(1, STATE_DIM+2);

  if (is_new_episode) {
    for (int i = 0; i < STACK_NUM; ++i) {
      free_matrix(stacked_frame[i]);
      stacked_frame[i] = matrix_clone(new_frame);
    }
    free_matrix(new_frame);
  } else {
    free_matrix(stacked_frame[0]);
    for (int i = 0; i < STACK_NUM-1; ++i) {
      stacked_frame[i] = stacked_frame[i+1];
    }
    stacked_frame[STACK_NUM-1] = new_frame;
  }

  for (int i = 0; i < STACK_NUM; ++i) {
    memcpy(ret->data+i*BASE_STATE_DIM, stacked_frame[i]->data, BASE_STATE_DIM*sizeof(float));
  }
  memcpy(ret->data+STACKED_BASE_STATE_DIM, last_full_state->data, last_full_state->cols*sizeof(float));

  return ret;
}

void run_ddpg_pixel() {
  // preparation phase
  last_full_state = new_matrix(1,1);
  stacked_frame = calloc(STACK_NUM, sizeof(matrix_t*));
  for (int i = 0; i < STACK_NUM; ++i) {
    stacked_frame[i] = new_matrix(1,1);
  }
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
  actor = init_model(STACKED_BASE_STATE_DIM);
  actor_target = init_model(STACKED_BASE_STATE_DIM);

  add_conv_layer(actor, 83, 83, 4, 32, 3, 2, 1, relu);
  add_max_pool_layer(actor, 42, 42, 32, 2, 2);
  add_conv_layer(actor, 21, 21, 32, 16, 3, 2, 0, relu);
  add_max_pool_layer(actor, 10, 10, 16, 2, 2);
  // add_conv_layer(actor, 42, 42, 3, 3, 3, 1, 1, relu);
  // add_max_pool_layer(actor, 42, 42, 3, 2, 2);

  // actor->output_dim++;
  // actor->max_out++;
  add_linear_layer(actor, 256, relu);
  add_linear_layer(actor, 64, relu);
  add_linear_layer(actor, ACTION_DIM, tanh_);

  matrix_t* W = actor->hidden_linears[2].data.l.W;
  for (int i = 0; i < W->cols*W->rows; ++i) {
    W->data[i] = rand_uniform(-0.003, 0.003);
  }


  add_conv_layer(actor_target, 83, 83, 4, 32, 3, 2, 1, relu);
  add_max_pool_layer(actor_target, 42, 42, 32, 2, 2);
  add_conv_layer(actor_target, 21, 21, 32, 16, 3, 2, 0, relu);
  add_max_pool_layer(actor_target, 10, 10, 16, 2, 2);
  // add_conv_layer(actor_target, 42, 42, 3, 3, 3, 1, 1, relu);
  // add_max_pool_layer(actor_target, 42, 42, 3, 2, 2);

  // actor_target->output_dim++;
  // actor->max_out++;
  add_linear_layer(actor_target, 256, relu);
  add_linear_layer(actor_target, 64, relu);
  add_linear_layer(actor_target, ACTION_DIM, tanh_);

  for (int i = 0; i < actor->num_of_layers; ++i) {
    linear_layer a = actor->hidden_linears[i].data.l;
    linear_layer a_target = actor_target->hidden_linears[i].data.l;

    copy_matrix(a_target.W, a.W);
    copy_matrix(a_target.b, a.b);
  }

  init_caches(actor, BATCH_SIZE);
  init_caches(actor_target, BATCH_SIZE);
  compile_model(actor, no_loss, adam);
  compile_model(actor_target, no_loss, adam);
  printf("actor max out %d actor target max out %d\n", actor->max_out, actor_target->max_out);
  return 1;
}

static int init_critic_w_target() {
  critic = init_model(AUX_STATE_DIM + ACTION_DIM);
  critic_target = init_model(AUX_STATE_DIM + ACTION_DIM);
  
  add_linear_layer(critic, 64, relu);
  add_linear_layer(critic, 64, relu);
  add_linear_layer(critic, 1, placeholder);

  add_linear_layer(critic_target, 64, relu);
  add_linear_layer(critic_target, 64, relu);
  add_linear_layer(critic_target, 1, placeholder);

  matrix_t* W = critic->hidden_linears[2].data.l.W;
  for (int i = 0; i < W->cols*W->rows; ++i) {
    W->data[i] = rand_uniform(-0.003, 0.003);
  }

  for (int i = 0; i < critic->num_of_layers; ++i) {
    linear_layer c = critic->hidden_linears[i].data.l;
    linear_layer c_target = critic_target->hidden_linears[i].data.l;

    copy_matrix(c_target.W, c.W);
    copy_matrix(c_target.b, c.b);
  }

  compile_model(critic, mse_loss, adam);
  compile_model(critic_target, mse_loss, adam);
  init_caches(critic, BATCH_SIZE);
  init_caches(critic_target, BATCH_SIZE);
  printf("critic max out %d critic target max out %d\n", critic->max_out, critic_target->max_out);
  return 1;
}

static int pre_training() {
  int is_new_episode = 1;
  matrix_t* state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST, RECV_STATE_DIM, ACTION_DIM);

  for (int i = 0; i < PRE_TRAIN_STEPS; ++i) {
    matrix_t* new_action = random_action(RECV_STATE_DIM, ACTION_DIM);//rand_uniform(-ACTION_BOUND, ACTION_BOUND);
    matrix_t* nxt_state = step(new_action, RECV_STATE_DIM, ACTION_DIM);

    state = stack_new_frame(state, is_new_episode);
    is_new_episode = 0;
    matrix_t* nxt_tmp = matrix_clone(nxt_state);
    matrix_t* nxt_stored_state = stack_new_frame(nxt_tmp, is_new_episode); 

    matrix_t* s = slice_col_wise(state, 0, STATE_DIM);
    matrix_t* s_a = concatenate(s, new_action, 1);
    matrix_t* s_a_ns = concatenate(s_a, nxt_stored_state, 1);

    store_experience(exp_buf, s_a_ns);
    free_matrix(s_a);
    free_matrix(s);
    free_matrix(new_action);
    free_matrix(nxt_stored_state);
    if (nxt_state->data[nxt_state->cols-2]) {
      is_new_episode = 1;
      free_matrix(nxt_state);
      free_matrix(state);
      state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST, RECV_STATE_DIM, ACTION_DIM);
    } else {
      // is_new_episode = 0;
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
  matrix_t* state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST, RECV_STATE_DIM, ACTION_DIM);
  int i;
  
  for (i = 0; i < MAX_EPOCH_LEN; ++i) {
    state = stack_new_frame(state, 1);
    // printf("getting action epd %d\n", i);
    matrix_t* new_action = get_action(state, NOISE_SCALE);
    // printf("getting next state\n");
    matrix_t* nxt_state = step(new_action, RECV_STATE_DIM, ACTION_DIM);

    matrix_t* nxt_tmp = matrix_clone(nxt_state);
    matrix_t* nxt_stored_state = stack_new_frame(nxt_tmp, 0);
    float new_reward = nxt_state->data[nxt_state->cols-1];
    sum += new_reward;
    matrix_t* s = slice_col_wise(state, 0, STATE_DIM);
    matrix_t* s_a = concatenate(s, new_action, 1);
    matrix_t* s_a_ns = concatenate(s_a, nxt_stored_state, 1);

    store_experience(exp_buf, s_a_ns);
    final_loss = train();
    free_matrix(s_a);
    free_matrix(s);
    free_matrix(nxt_stored_state);
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
  // augment_space(action, action->rows, actor->max_out);
  // predict(actor, action);
  // printf("state row %d col %d\n", action->rows, action->cols);
  modified_predict(actor, action);
  // predict(actor, action);
  mult_scalar(action, ACTION_BOUND);
  matrix_t* noise = rand_normal(ACTION_DIM);
  mult_scalar(noise, noise_scale);
  elem_wise_add(action, noise);
  free_matrix(noise);
  clip(action, -ACTION_BOUND, ACTION_BOUND);
  return action;
}

static int modified_predict(model* m, matrix_t* input) {
  // assert(x->rows > 0 && x->cols > 0);
  assert(input->cols == STATE_DIM);
  matrix_t* x = slice_col_wise(input, 0, input->cols-AUX_STATE_DIM);

  // float angular = input->data[input->cols-1];
  // matrix_t* obs = slice_col_wise(input, input->cols-AUX_STATE_DIM, input->cols);
  // augment_space(x, x->rows, m->max_out);
  // augment_space(input, input->rows, m->max_out);
  if (!m->cache_initialized) {
    printf("[CAVEAT] Automatically initializing cache with size %d\n", x->rows);
    init_caches(m, x->rows);
  }

  // forward(m->hidden_linears, x);
  // forward(m->hidden_activations, x);
  // forward(m->)
  // // x->cols++;
  // // x->data[x->cols-1] = angular;
  // // memcpy(x->data+x->cols, obs->data, obs->cols*sizeof(float));
  // // x->cols++;
  // // print_matrix(x,1);

  // forward(m->hidden_linears+1, x);
  // forward(m->hidden_activations+1, x);
  // forward(m->hidden_linears+2, x);
  // forward(m->hidden_activations+2, x);

  predict(m, x);
  copy_matrix(input, x);
  free_matrix(x);
  // free_matrix(obs);
  return 1;
}


static float train() {
  // training on past experiences
  matrix_t* batch = sample_experience(exp_buf, BATCH_SIZE);
  matrix_t* states = slice_col_wise(batch, 0, STATE_DIM);
  matrix_t* actions = slice_col_wise(batch, STATE_DIM, STATE_DIM+ACTION_DIM);
  matrix_t* nxt_states = slice_col_wise(batch, STATE_DIM+ACTION_DIM, STATE_DIM+STATE_DIM+ACTION_DIM);
  // printf("nxt_states rows %d cols %d start %d end %d\n", nxt_states->rows, nxt_states->cols, STATE_DIM+ACTION_DIM, sSTATE_DIM+ACTION_DIM);
  matrix_t* dones = slice_col_wise(batch, STATE_DIM+STATE_DIM+ACTION_DIM, STATE_DIM+STATE_DIM+ACTION_DIM+1);
  matrix_t* rewards = slice_col_wise(batch, STATE_DIM+STATE_DIM+ACTION_DIM+1, STATE_DIM+STATE_DIM+ACTION_DIM+2);

  // states = slice_row_wise(states, 0, 1);
  // save_image(slice_col_wise(states,0,BASE_STATE_DIM), 83, 83, "state");

  // save_image(slice_col_wise(states,BASE_STATE_DIM, 2*BASE_STATE_DIM), 83, 83, "state1");
  // save_image(slice_col_wise(states,BASE_STATE_DIM*2,3*BASE_STATE_DIM), 83, 83, "state2");
  // save_image(slice_col_wise(states,BASE_STATE_DIM*3,4*BASE_STATE_DIM), 83, 83, "state3");
  // nxt_states = slice_row_wise(nxt_states, 0, 1);
  // save_image(slice_col_wise(nxt_states, 0, BASE_STATE_DIM), 83, 83, "nxt state");
  // save_image(slice_col_wise(nxt_states, BASE_STATE_DIM, 2*BASE_STATE_DIM), 83, 83, "nxt state1");
  // save_image(slice_col_wise(nxt_states, BASE_STATE_DIM*2, 3*BASE_STATE_DIM), 83, 83, "nxt state2");
  // save_image(slice_col_wise(nxt_states, BASE_STATE_DIM*3, 4*BASE_STATE_DIM), 83, 83, "nxt state3");
  // states = slice_col_wise(states, 4*BASE_STATE_DIM, states->cols);
  // nxt_states =slice_col_wise(nxt_states, 4*BASE_STATE_DIM, nxt_states->cols);
  // // printf("state rows %d cols %d\n", states->rows, states->cols);
  // // printf("nxt state rows %d cols %d\n", nxt_states->rows, nxt)
  // print_matrix(states,1);
  // print_matrix(nxt_states,1);

  // exit(1);

  // printf("states %d %d\n", states->rows, states->cols);
  // printf("actions %d %d\n", actions->rows, actions->cols);
  // printf("nxt_states %d %d\n", nxt_states->rows, nxt_states->cols);
  // printf("dones %d %d\n", dones->rows, dones->cols);
  // printf("rewards %d %d\n", rewards->rows, rewards->cols);
  // exit(1);

  // calculating critic's target
  matrix_t* nxt_actions = matrix_clone(nxt_states);
  // printf("nxt states rows %d cols %d STATE_DIM %d batch rows %d cols %d\n", nxt_actions->rows, nxt_actions->cols, STATE_DIM, batch->rows, batch->cols);
  modified_predict(actor_target, nxt_actions);
  // predict(actor_target, nxt_actions);
  matrix_t* nxt_full_states = slice_col_wise(nxt_states, STACKED_BASE_STATE_DIM, STATE_DIM);

  matrix_t* nxt_qs = concatenate(nxt_full_states, nxt_actions, 1);
  

  predict(critic_target, nxt_qs);
  neg(dones);
  add_scalar(dones, 1);
  elem_wise_mult(nxt_qs, dones);
  mult_scalar(nxt_qs, GAMMA);
  elem_wise_add(rewards, nxt_qs);
  // update critic
  matrix_t* full_state = slice_col_wise(states, STACKED_BASE_STATE_DIM, STATE_DIM);
  matrix_t* qs = concatenate(full_state, actions, 1);
  fit(critic, qs, rewards, BATCH_SIZE, 1, C_LR, 0, 1);

  // find gradient of Q w.r.t action
  matrix_t* n_actions = matrix_clone(states);
  modified_predict(actor, n_actions);
  // predict(actor, n_actions);
  matrix_t* q_n_actions = concatenate(full_state, n_actions, 1);
  predict(critic, q_n_actions);
  float final_loss = mean(q_n_actions);
  matrix_t* c_grad = matrix_clone(q_n_actions);
  for (int i = 0; i < c_grad->rows*c_grad->cols; ++i) {
    c_grad->data[i] = 1 / (float)c_grad->rows;
  }

  // mult_scalar(c_grad, 1/(float)c_grad->rows);
  model_backward(critic, c_grad);
  assert(c_grad->cols == AUX_STATE_DIM + ACTION_DIM);
  matrix_t* a_grad = slice_col_wise(c_grad, AUX_STATE_DIM, AUX_STATE_DIM+ACTION_DIM);
  mult_scalar(a_grad, ACTION_BOUND);
  // mult_scalar(a_grad, 1/(float) a_grad->rows);
  neg(a_grad);

  // back propagation and update
  model_backward(actor, a_grad);
  // augment_space(a_grad, a_grad->rows, actor->max_out);
  // backward(actor->hidden_activations+2, a_grad);
  // backward(actor->hidden_linears+2, a_grad);
  // backward(actor->hidden_activations+1, a_grad);
  // backward(actor->hidden_linears+1, a_grad);

  // a_grad->cols--;

  // backward(actor->hidden_activations, a_grad);
  // backward(actor->hidden_linears, a_grad);

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
  free_matrix(full_state);
  free_matrix(nxt_full_states);
  return final_loss;
}

static void save_all_model() {
  save_model(actor, DDPG_ACTOR_FILE);
  save_model(actor_target, DDPG_ACTOR_T_FILE);
  save_model(critic, DDPG_CRITIC_FILE);
  save_model(critic_target, DDPG_CRITIC_T_FILE);
}
