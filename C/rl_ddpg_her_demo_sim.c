#include "rl_ddpg_her_demo_sim.h"
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
#include "normalizer.h"
#include "macros.h"
#ifdef MPI
#include "multi_agents/mpi_utils.h"
#endif

#define STATE_DIM 14 //16
#define G_DIM 3
#define AG_DIM 3
#define ACTION_DIM 4
#define GAMMA 0.98
#define C_LR 0.001
#define A_LR 0.001
#define EPOCH 100000
#define POLYAK 0.95
#define MAX_EPOCH_LEN 1000
#define BATCH_SIZE 1024 // same as 64 timesteps
#define PRE_TRAIN_STEPS 0
#define MEMORY_SIZE 1000000
#define NOISE_SCALE 0.1
#define RANDOM_INIT_ANGLE 0
#define RANDOM_INIT_DEST 1
#define NUM_OF_LAYERS 4
#define ACTION_BOUND 1
#define ENV_LIMIT 50
#define PORTION_OF_TRANSITION_WITH_ADDITIONAL_GOAL 0.8
#define REPLAY_K 4
#define N_BATCHES 40
#define RANDOM_EPS 0.1
#define OBS_CLIP_RANGE 5
#define NORMALIZE 1
#define Q_RANGE ENV_LIMIT
#define PRM_LOSS_WEIGHT 0.001
#define AUX_LOSS_WEIGHT 0.0078
#define NUM_DEMO 100
#define DEMO_BATCH_SIZE 128 // 32/256 or 128/1024
#define NUM_DEMO_TRANSITIONS 5000

#define DDPG_ACTOR_FILE "DDPG_ACTOR_PICKNPLACE_NORM_SIM.model"
#define DDPG_ACTOR_T_FILE "DDPG_ACTOR_T_PICKNPLACE_NORM_SIM.model"
#define DDPG_CRITIC_FILE "DDPG_CRITIC_PICKNPLACE_NORM_SIM.model"
#define DDPG_CRITIC_T_FILE "DDPG_CRITI_T_PICKNPLACE_NORM_SIM.model"
#define DDPG_NORM_FILE "DDPG_NORM_PICKNPLACE_NORM_SIM.norm"


static int actor_layers_config[NUM_OF_LAYERS] = {256, 256, 256, ACTION_DIM};
static int critic_layers_config[NUM_OF_LAYERS] = {256, 256, 256, 1};
static layer_type actor_layers_acts[NUM_OF_LAYERS] = {relu, relu, relu, tanh_};
static layer_type critic_layers_acts[NUM_OF_LAYERS] = {relu, relu, relu, placeholder};

static model *actor, *critic, *actor_target, *critic_target;
static experience_buffer* exp_buf;
static experience_buffer* demo_buf;
static normalizer* norm;

static int store_sample_her(experience_buffer* expbuf, matrix_t** episode_s1, matrix_t** episode_s2, matrix_t** episode_a, int count);
static int init_actor_w_target();
static int init_critic_w_target();
static int pre_training();
//  double reward(matrix_t* last, matrix_t* curr);
static matrix_t* get_action(matrix_t* state, double act_noise);
static double* run_epoch();
static double* train();
static void save_all_model();
static int update_target();
#ifdef MPI
static int rank;
#endif

void run_ddpg_her_w_demo_sim() {
  // preparation phase
  #ifdef MPI
  rank = mpi_init();
  #endif
  init_actor_w_target();
  init_critic_w_target();
  demo_buf = init_demo_buffer(NUM_DEMO, STATE_DIM+ACTION_DIM+STATE_DIM+2);
  exp_buf = init_experience_buffer(MEMORY_SIZE);
  if (NORMALIZE) {
    norm = init_normalizer(STATE_DIM, DEFAULT_CLIP_RANGE);
    matrix_t* demos = sample_experience(demo_buf, NUM_DEMO_TRANSITIONS);
    matrix_t** separated = calloc(NUM_DEMO_TRANSITIONS, sizeof(*separated));
    for (int i = 0; i < NUM_DEMO_TRANSITIONS; ++i) {
      separated[i] = slice_row_wise(demos, i, i+1);
    }
    update_normalizer(norm, separated, NUM_DEMO_TRANSITIONS);
    free_matrix(demos);
    for (int i = 0; i < NUM_DEMO_TRANSITIONS; ++i) {
      free_matrix(separated[i]);
    }
    free(separated);
  }
  initEnv(ACTION_DIM, ENV_PICK_N_PLACE);
  printf("Initialized models\n");
  // randomly explore for certain number of steps
  if (!pre_training()) {
    printf("[RUN_DDPG_HER] failed to fill the experience buffer");
  }
  printf("Done pre-training\n");

  // training
  int epc = 0;
  clock_t start = clock(), diff;
  while (epc < EPOCH) {
    epc++;
    double* info = run_epoch();
    // printf("sampled\n");
    double* train_info = NULL;
    for (int i = 0; i < N_BATCHES; ++i) {
      train_info = train();
      if (i != N_BATCHES - 1) {
        free(train_info);
      }
    }
    update_target();
    // printf("trained\n");
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    #ifdef MPI
    if (!rank) {
      printf("Episode: %d | Rewards: %.3f | Critic_loss: %.1f | Mean Q: %.1f| Time elapsed: %.1f mins \n", epc, info[0], train_info[0], train_info[1],msec/(double)60000);
    }
    #else
    printf("Episode: %d | Rewards: %.3f | Critic_loss: %.1f | Mean Q: %.1f| Time elapsed: %.1f mins \n", epc, info[0], train_info[0], train_info[1],msec/(double)60000);
    #endif
    free(info);
    free(train_info);
    if (epc % 1000 == 0) {
      save_all_model();
    }
  }
  
  closeEnv(STATE_DIM+AG_DIM, ACTION_DIM);

  free_experience_buffer(exp_buf);
  free_experience_buffer(demo_buf);
  free_model(actor);
  free_model(actor_target);
  free_model(critic);
  free_model(critic_target);
  #ifdef MPI
  mpi_finalize();
  #endif
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
    // if (i == NUM_OF_LAYERS-1) {
    //   for (int i = 0; i < W->cols*W->rows; ++i) {
    //     W->data[i] = rand_uniform(-0.003, 0.003);
    //   }
    // }
    copy_matrix(W_target, W);
    copy_matrix(b_target, b);
  }
  init_caches(actor, BATCH_SIZE);
  init_caches(actor_target, BATCH_SIZE);
  compile_model(actor, no_loss, adam);
  compile_model(actor_target, no_loss, adam);
  #ifdef MPI
  mpi_sync(actor);
  mpi_sync(actor_target);
  #endif
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
    // if (i == NUM_OF_LAYERS-1) {
    //   for (int i = 0; i < W->cols*W->rows; ++i) {
    //     W->data[i] = rand_uniform(-0.003, 0.003);
    //   }
    // }
    copy_matrix(W_target, W);
    copy_matrix(b_target, b);
  }
  compile_model(critic, mse_loss, adam);
  compile_model(critic_target, mse_loss, adam);
  init_caches(critic, BATCH_SIZE);
  init_caches(critic_target, BATCH_SIZE);
  #ifdef MPI
  mpi_sync(critic);
  mpi_sync(critic_target);
  #endif
  return 1;
}

static int pre_training() {
  matrix_t* state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST, STATE_DIM+AG_DIM, ACTION_DIM);
  for (int i = 0; i < PRE_TRAIN_STEPS; ++i) {
    matrix_t* new_action = random_action(STATE_DIM+AG_DIM, ACTION_DIM);//rand_uniform(-ACTION_BOUND, ACTION_BOUND);
    matrix_t* nxt_state = step(new_action, STATE_DIM+AG_DIM, ACTION_DIM);
    //print_matrix(nxt_state, 1);
    matrix_t* nxt_observation = slice_col_wise(nxt_state, 0, STATE_DIM);
    matrix_t* other_info = slice_col_wise(nxt_state, STATE_DIM+AG_DIM, nxt_state->cols);
    //free_matrix(nxt_state);
    matrix_t* temp = concatenate(nxt_observation, other_info, 1);
    free_matrix(nxt_observation);
    free_matrix(other_info);
    matrix_t* s = slice_col_wise(state, 0, STATE_DIM);
    matrix_t* s_a = concatenate(s, new_action, 1);
    matrix_t* s_a_ns = concatenate(s_a, temp, 1);

    store_experience(exp_buf, s_a_ns);
    free_matrix(s_a);
    free_matrix(s);
    free_matrix(new_action);
    free_matrix(temp);
    if (nxt_state->data[nxt_state->cols-2]) {
      free_matrix(nxt_state);
      free_matrix(state);
      state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST, STATE_DIM+AG_DIM, ACTION_DIM);
    } else {
      free_matrix(state);
      state = nxt_state;
    }
  }
  free_matrix(state);
  return 1;
}

static double* run_epoch() {
  double sum = 0;
  double dones = 0;
  //double final_loss = 0;
  matrix_t** episode_s1 = calloc(MAX_EPOCH_LEN, sizeof(matrix_t*));
  matrix_t** episode_s2 = calloc(MAX_EPOCH_LEN, sizeof(matrix_t*));
  matrix_t** episode_a = calloc(MAX_EPOCH_LEN, sizeof(matrix_t*));

  if (!episode_a || !episode_s1 || !episode_s2) {
    printf("[RUN_EPOCH] malloc failed\n");
    exit(1);
  }

  matrix_t* state = resetState(RANDOM_INIT_ANGLE, RANDOM_INIT_DEST, STATE_DIM+AG_DIM, ACTION_DIM);

  int count = 0;
  for (int i = 0; i < MAX_EPOCH_LEN; ++i) {
    matrix_t* new_action = get_action(state, NOISE_SCALE);
    matrix_t* nxt_state = step(new_action, STATE_DIM+AG_DIM, ACTION_DIM);
    double new_reward = nxt_state->data[nxt_state->cols-1];
    sum += new_reward;
  
    episode_s1[i] = matrix_clone(state);
    episode_s2[i] = matrix_clone(nxt_state);
    episode_a[i] = matrix_clone(new_action);
    count++;
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

  if (NORMALIZE) {
    #ifdef MPI
    mpi_update_normalizer(norm, episode_s1, count);
    #else
    update_normalizer(norm, episode_s1, count);
    #endif
  }
  //update_normalizer(norm, episode_s2, count);
  store_sample_her(exp_buf, episode_s1, episode_s2, episode_a, count);

  double* ret = calloc(2, sizeof(double));
  ret[0] = sum;
  ret[1] = dones;
  return ret;
}

static int store_sample_her(experience_buffer* exp_buf, matrix_t** episode_s1, matrix_t** episode_s2, matrix_t** episode_a, int count) {
  double add_replay = PORTION_OF_TRANSITION_WITH_ADDITIONAL_GOAL;
  for (int i = 0; i < count; ++i) {
    // printf("doing %d\n", i);
    matrix_t* nxt_s1 = episode_s1[i];
    matrix_t* nxt_s2 = episode_s2[i];
    matrix_t* nxt_a = episode_a[i];
 
    matrix_t* nxt_ob1 = slice_col_wise(nxt_s1, 0, STATE_DIM);
    matrix_t* nxt_ob2 = slice_col_wise(nxt_s2, 0, STATE_DIM);
    matrix_t* nxt_other_info = slice_col_wise(nxt_s2, STATE_DIM+AG_DIM, nxt_s2->cols);

    matrix_t* s_a = concatenate(nxt_ob1, nxt_a, 1);
    matrix_t* s_a_ns = concatenate(s_a, nxt_ob2, 1);
    matrix_t* s_a_ns_dr = concatenate(s_a_ns, nxt_other_info, 1);

    store_experience(exp_buf, s_a_ns_dr);

    if (rand_uniform(0, 1) < add_replay) {
      for (int k = 0; k < REPLAY_K; ++k) {
        int future_idx = (int) rand_uniform(i, count-1);
        matrix_t* future_ob1 = episode_s1[future_idx];
        matrix_t* future_ob2 = episode_s2[future_idx];
        matrix_t* ag;
        double new_reward = future_idx-i<=1 ? 0 : -1;
        if (future_idx == i) {
          ag = slice_col_wise(future_ob2, STATE_DIM, STATE_DIM+AG_DIM);
        } else {
          ag = slice_col_wise(future_ob1, STATE_DIM, STATE_DIM+AG_DIM);
        }
        matrix_t* s_a_ns_dr_additional = matrix_clone(s_a_ns_dr);
        int goal_offset_1 = STATE_DIM - G_DIM;
        int goal_offset_2 = STATE_DIM + ACTION_DIM + STATE_DIM - G_DIM;
        memcpy(s_a_ns_dr_additional->data+goal_offset_1, ag->data, G_DIM*sizeof(double));
        memcpy(s_a_ns_dr_additional->data+goal_offset_2, ag->data, G_DIM*sizeof(double));
        s_a_ns_dr_additional->data[STATE_DIM*2+ACTION_DIM+1] = new_reward;
        store_experience(exp_buf, s_a_ns_dr_additional);
        free_matrix(ag);
      }
    }
    free_matrix(s_a_ns);
    free_matrix(s_a);
    free_matrix(nxt_other_info);
    free_matrix(nxt_ob2);
    free_matrix(nxt_ob1);
    free_matrix(nxt_a);
    free_matrix(nxt_s2);
    free_matrix(nxt_s1);
  }
  free(episode_s1);
  free(episode_s2);
  free(episode_a);

  return 1;
}

static matrix_t* get_action(matrix_t* state, double noise_scale) {
  if (rand_uniform(0, 1) < RANDOM_EPS) {
    return random_action(STATE_DIM+AG_DIM, ACTION_DIM);
  }
  matrix_t* action = slice_col_wise(state, 0, STATE_DIM);
  if (NORMALIZE) {
    normalize_obs(norm, action);
  }
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


static double* train() {
  // training on past experiences, demo experiences, BC loss, Q filter (l2 regularize on action)
  matrix_t* policy_batch = sample_experience(exp_buf, BATCH_SIZE-DEMO_BATCH_SIZE);
  matrix_t* demo_batch = sample_experience(demo_buf, DEMO_BATCH_SIZE);
  //printf("concatenating\n");
  matrix_t* batch = concatenate(policy_batch, demo_batch, 0);
  //printf("concatenated\n");
  matrix_t* states = slice_col_wise(batch, 0, STATE_DIM);
  matrix_t* actions = slice_col_wise(batch, STATE_DIM, STATE_DIM+ACTION_DIM);
  matrix_t* nxt_states = slice_col_wise(batch, STATE_DIM+ACTION_DIM, 2*STATE_DIM+ACTION_DIM);
  matrix_t* dones = slice_col_wise(batch, 2*STATE_DIM+ACTION_DIM, 2*STATE_DIM+ACTION_DIM+1);
  matrix_t* rewards = slice_col_wise(batch, 2*STATE_DIM+ACTION_DIM+1, 2*STATE_DIM+ACTION_DIM+2);

  
  matrix_t* demo_states = slice_col_wise(demo_batch, 0, STATE_DIM);
  matrix_t* demo_actions = slice_col_wise(demo_batch, STATE_DIM, STATE_DIM+ACTION_DIM);
  // matrix_t* demo_nxt_states = slice_col_wise(demo_batch, STATE_DIM+ACTION_DIM, 2*STATE_DIM+ACTION_DIM);
  // matrix_t* demo_dones = slice_col_wise(demo_batch, 2*STATE_DIM+ACTION_DIM, 2*STATE_DIM+ACTION_DIM+1);
  // matrix_t* demo_rewards = slice_col_wise(demo_batch, 2*STATE_DIM+ACTION_DIM+1, 2*STATE_DIM+ACTION_DIM+2);
  
  if (NORMALIZE) {
    normalize_obs(norm, demo_states);
    normalize_obs(norm, states);
    normalize_obs(norm, nxt_states);
  }
  //printf("normalized for training\n");
  // calculating critic's target
  matrix_t* nxt_actions = matrix_clone(nxt_states);
  predict(actor_target, nxt_actions);
  matrix_t* nxt_qs = concatenate(nxt_states, nxt_actions, 1);
  
  predict(critic_target, nxt_qs);
  // neg(dones);
  // add_scalar(dones, 1);
  // elem_wise_mult(nxt_qs, dones);
  mult_scalar(nxt_qs, GAMMA);
  elem_wise_add(rewards, nxt_qs);
  clip(rewards, -Q_RANGE, 0);

  // update critic
  matrix_t* qs = concatenate(states, actions, 1);
  //printf("updated critic\n");

  // find gradient of Q w.r.t action
  matrix_t* n_actions = matrix_clone(states);
  predict(actor, n_actions);
  matrix_t* q_n_actions = concatenate(states, n_actions, 1);
  predict(critic, q_n_actions);
  matrix_t* c_grad = matrix_clone(q_n_actions);
  double mean_q = mean(q_n_actions);
  for (int i = 0; i < c_grad->rows*c_grad->cols; ++i) {
    c_grad->data[i] = 1 / (double)c_grad->rows;
  }

  model_backward(critic, c_grad);
  assert(c_grad->cols == STATE_DIM + ACTION_DIM);
  matrix_t* a_grad = slice_col_wise(c_grad, STATE_DIM, STATE_DIM+ACTION_DIM);
  mult_scalar(a_grad, ACTION_BOUND);
  neg(a_grad);
  mult_scalar(a_grad, PRM_LOSS_WEIGHT);
  //printf("found gradient of q w.r.t action\n");


  // Q filter and add bc loss (l2 regularizer)
  int demo_offset = BATCH_SIZE - DEMO_BATCH_SIZE;
  matrix_t* demo_q = concatenate(demo_states, demo_actions, 1);
  predict(critic, demo_q);
  matrix_t* policy_act_demo = matrix_clone(demo_states);
  predict(actor, policy_act_demo);
  matrix_t* policy_act_demo_target = matrix_clone(demo_actions);
  matrix_t* bc_grad = matrix_clone(policy_act_demo);
  elem_wise_minus(bc_grad, policy_act_demo_target);
  mult_scalar(bc_grad, (double)2);
  //mult_scalar(bc_grad, 1/(double)bc_grad->rows);
  for (int i = 0; i < DEMO_BATCH_SIZE; ++i) {
    if (q_n_actions->data[i+demo_offset] > demo_q->data[i]) {
      for (int j = 0; j < bc_grad->cols; ++j) {
        bc_grad->data[i*bc_grad->cols+j] = 0;
      }
    }
  }
  mult_scalar(bc_grad, AUX_LOSS_WEIGHT);
  matrix_t* padding = new_matrix(a_grad->rows-bc_grad->rows, a_grad->cols);
  initialize(padding, zeros);
  matrix_t* bc_grad_padded = concatenate(padding, bc_grad, 0);
  //printf("found bc grad\n");
  elem_wise_add(a_grad, bc_grad_padded);

  //refresh cache
  matrix_t* refresher = matrix_clone(states);
  predict(actor, refresher);

  // back propagation and update
  #ifdef MPI
  double final_loss = fit(critic, qs, rewards, BATCH_SIZE, 1, C_LR, 0, 0);
  mpi_perform_update(critic, C_LR, 0);
  #else
  double final_loss = fit(critic, qs, rewards, BATCH_SIZE, 1, C_LR, 0, 1);
  #endif
  model_backward(actor, a_grad);
  #ifdef MPI
  mpi_perform_update(actor, A_LR, 0);
  #else
  perform_update(actor, A_LR);
  #endif
  //printf("freeing resource\n");
  free_matrix(refresher);
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
  free_matrix(policy_batch);
  free_matrix(demo_batch);
  free_matrix(demo_states);
  free_matrix(demo_actions);
  free_matrix(demo_q);
  free_matrix(policy_act_demo);
  free_matrix(policy_act_demo_target);
  free_matrix(bc_grad);
  free_matrix(padding);
  free_matrix(bc_grad_padded);

  double* info = calloc(2, sizeof(double));
  info[0] = final_loss;
  info[1] = mean_q;
  return info;
}

static int update_target() {
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
  return 1;
}

static void save_all_model() {
  save_model(actor, DDPG_ACTOR_FILE);
  save_model(actor_target, DDPG_ACTOR_T_FILE);
  save_model(critic, DDPG_CRITIC_FILE);
  save_model(critic_target, DDPG_CRITIC_T_FILE);
  if (NORMALIZE) {
    save_normalizer(norm, DDPG_NORM_FILE);
  }
}



