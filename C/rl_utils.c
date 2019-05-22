#include "rl_utils.h"
#include <stdlib.h>
#include <assert.h>
#include "utils.h"
#include <string.h>
#include <stdio.h>
#include "socket_utils.h"
#include "sim_api.h"
#include <math.h>

#ifdef C_AS_LIB
static experience_buffer* build_sim_demo_buffer(int size, int transition_dim);
#endif

experience_buffer* init_experience_buffer(int size) {
  assert(size > 0);
  experience_buffer* new_buffer = (experience_buffer*)malloc(sizeof(experience_buffer));
  new_buffer->experiences = calloc(size, sizeof(matrix_t*));
  new_buffer->max_size = size;
  new_buffer->curr_size = 0;
  new_buffer->end_pos = 0;
  return new_buffer;
}

experience_buffer* init_demo_buffer(int size, int transition_dim) {
  #ifndef C_AS_LIB
  init_demo_connection();
  return build_demo_buffer(size, transition_dim);
  #else
  return build_sim_demo_buffer();
  #endif
}

int store_experience(experience_buffer* exp_buf, matrix_t* new_exp) {
  if (exp_buf->curr_size >= exp_buf->max_size) {
    free_matrix(exp_buf->experiences[exp_buf->end_pos]);
  } else {
    exp_buf->curr_size++;
  }
  exp_buf->experiences[exp_buf->end_pos] = new_exp;
  exp_buf->end_pos = (exp_buf->end_pos + 1) % exp_buf->max_size;
  
  return 1;
}

int free_experience_buffer(experience_buffer* exp_buf) {
  for (int i = 0; i < exp_buf->curr_size; ++i) {
    free_matrix(exp_buf->experiences[i]);
  }
  free(exp_buf->experiences);
  free(exp_buf);
  return 1;
}

matrix_t* sample_experience(experience_buffer* exp_buf, int num) {
  assert(num <= exp_buf->max_size);
  if (exp_buf->curr_size < num) {
    num = exp_buf->curr_size;
  }
  matrix_t* ret = new_matrix(num, exp_buf->experiences[0]->cols);
  
  int idx;
  int chosen[exp_buf->curr_size];
  memset(chosen, 0, exp_buf->curr_size*sizeof(int));
  for (int i = 0; i < num; ++i) {
    idx = (int) rand_uniform(-1, exp_buf->curr_size);
    if (idx == -1) idx++;
    if (idx == exp_buf->curr_size) idx--;
    for (int j = idx; j < exp_buf->curr_size; ++j) {
      if (!chosen[j]) {
        idx = j;
        chosen[j] = 1;
        break;
      }
    }
    assert(exp_buf->experiences[idx]->cols == ret->cols);
    memcpy(ret->data+(i*ret->cols), exp_buf->experiences[idx]->data, ret->cols*sizeof(double));
  }

  return ret;
}

void print_experiences(experience_buffer* exp_buf) {
  assert(exp_buf->curr_size > 0);
  matrix_t* all = new_matrix(exp_buf->curr_size, exp_buf->experiences[0]->cols);
  for (int i = 0; i < exp_buf->curr_size; ++i) {
    memcpy(all->data+(i*all->cols), exp_buf->experiences[i]->data, all->cols*sizeof(double));
  }
  print_matrix(all, 1);
}

#ifdef C_AS_LIB

static matrix_t** go_to_point(matrix_t* obs, matrix_t* pos, int* timestep);
static matrix_t** fill_up_episode(matrix_t* obs, int curr_time_step, int to_time_step);
static double distance(matrix_t* a, matrix_t* b);
static matrix_t* normalize_action(matrix_t* action);

static int reward_dim = 1;
static int info_dim = 1;
static int act_dim = 4;
static int g_dim = 3;
static int obj_pos_offset = 6;
static int state_dim = 14;
static double dist_threshold = 0.05;

static int rand_angle = 0;
static int rand_dest_pos = 1;
static double act_clip_range = 0.0872665; // 5 degrees
static int env_step_limit = 50;

static experience_buffer* build_sim_demo_buffer(int size, int transition_dim) {
  assert(transition_dim == 2*state_dim + act_dim + info_dim + reward_dim);
  experience_buffer* ret = init_experience_buffer(size);
  
  printf("Building buffer with state dimension: %d, action dimension: %d, \
            total transition dimension: %d\n", state_dim, act_dim, transition_dim);

  initEnv(0, ENV_PICK_N_PLACE);

  for (int i = 0; i < size; ++i) {
    matrix_t* init_obs = resetState(rand_angle, rand_dest_pos, 0, 0);
    int time_step = 0;
    // Reach the target with end effector
    matrix_t* first_stop = slice_col_wise(init_obs, obj_pos_offset, obj_pos_offset+g_dim);
    matrix_t** transitions = go_to_point(init_obs, first_stop, &time_step);
    if (time_step > env_step_limit) {
      printf("Too many steps taken to reach the target, resampling target\n");
      i--;
      continue;
    }
    for (int j = 0; j < time_step; ++j) {
      store_experience(ret, transitions[i]);
    }
    // Grab up the target
    matrix_t* last_transition = transitions[time_step-1];
    matrix_t* last_obs = slice_col_wise(last_transition, state_dim+act_dim, state_dim+act_dim+state_dim);
    matrix_t* grab_act = new_matrix(1, act_dim);
    initialize(grab_act, zeros);
    grab_act->data[grab_act->cols-1] = 1;
    matrix_t* nxt_state = step(grab_act, 0, 0);
    matrix_t* nxt_obs = slice_col_wise(nxt_state, 0, state_dim);
    matrix_t* other_info = slice_col_wise(nxt_state, state_dim+g_dim, nxt_state->cols);
    matrix_t* o_a = concatenate(last_obs, grab_act, 1);
    matrix_t* o_a_o2 = concatenate(o_a, nxt_obs, 1);
    matrix_t* o_a_o2_dr = concatenate(o_a_o2, other_info, 1);
    time_step++;
    if (time_step > env_step_limit) {
      printf("Too many steps taken to grab the target, resampling target\n");
      i--;
      continue;
    }
    store_experience(ret, o_a_o2_dr);
    // Deliver the target
    matrix_t* nxt_stop = slice_col_wise(nxt_state, state_dim-g_dim, state_dim);
    matrix_t** phase_2_transition = go_to_point(nxt_state ,nxt_stop, &time_step);
    if (time_step > env_step_limit) {
      printf("Too many steps taken to deliver the target, resampling target\n");
      i--;
      continue;
    }
    for (int j = 0; j < time_step; ++j) {
      store_experience(ret, phase_2_transition[i]);
    }
    // Fill up the episode with padding
    matrix_t* phase_2_last_transition = phase_2_transition[time_step-1];
    matrix_t* phase_2_last_obs = slice_col_wise(phase_2_last_transition, state_dim+act_dim, phase_2_last_transition->cols);
    matrix_t** padding_transitions = fill_up_episode(phase_2_last_obs, time_step, env_step_limit);
    for (int j = 0; j < env_step_limit; ++j) {
      store_experience(ret, padding_transitions[i]);
    }

    // check behaviour with rendering
    matrix_t** all_actions = calloc(env_step_limit, sizeof(*all_actions));
    for (int j = 0; j < env_step_limit; ++j) {
      int offset = i * env_step_limit;
      matrix_t* nxt_act = slice_col_wise(ret->experiences[j+offset], state_dim, state_dim+act_dim);
      all_actions[j] = nxt_act;
      renderSteps(all_actions, env_step_limit);
      // free resources

    }

    // free resources

  }

  return ret;
}

static matrix_t** go_to_point(matrix_t* obs, matrix_t* pos, int* timestep) {
  //observation: joint angles(3), ee position(3), ee state(1), object position(3), has object(1)
  //target position (dest position): (3)
  matrix_t* state = slice_col_wise(obs, 0, state_dim);
  matrix_t* state_ee_pos = slice_col_wise(state, g_dim, 2*g_dim);
  matrix_t** ret = calloc(env_step_limit, sizeof(*ret));
  while(distance(state_ee_pos, pos) > dist_threshold) {
    matrix_t* nxt_action = inverse_km(pos);
    clip(nxt_action, -act_clip_range, act_clip_range);
    matrix_t* normed_action = normalize_action(nxt_action);
    clip(normed_action, -1, 1);
    matrix_t* nxt_obs = step(normed_action, 0, 0);
    matrix_t* o2 = slice_col_wise(nxt_obs, 0, state_dim);
    matrix_t* dr = slice_col_wise(nxt_obs, state_dim+g_dim, nxt_obs->cols);
    
    matrix_t* o_a = concatenate(state, normed_action, 1);
    matrix_t* o_a_o2 = concatenate(o_a, o2, 1);
    matrix_t* o_a_o2_dr = concatenate(o_a_o2, dr, 1);

    ret[*timestep] = o_a_o2_dr;
    *timestep += 1;
  }
  return ret;
}

static matrix_t** fill_up_episode(matrix_t* obs, int curr_time_step, int to_time_step) {
  matrix_t** ret = calloc(to_time_step-curr_time_step, sizeof(*ret));
  matrix_t* state = slice_col_wise(obs, 0, state_dim);
  matrix_t* action = new_matrix(1, act_dim);
  initialize(action, zeros);
  for (int i = curr_time_step; i < to_time_step; ++i) {
    matrix_t* o_a = concatenate(state, action, 1);
    matrix_t* o_a_o2_dr = concatenate(o_a, obs, 1);
    ret[i-curr_time_step] = o_a_o2_dr;
  }
  return ret;
}

static double distance(matrix_t* a, matrix_t* b) {
  assert(a->rows == b->rows);
  assert(a->rows == 1);
  matrix_t* temp = matrix_clone(a);
  elem_wise_minus(temp, b);
  elem_wise_mult(temp, temp);
  double sum = 0;
  for (int i = 0; i < a->cols; ++i) {
    sum += temp->data[i];
  }
  return sqrt(sum);
}

static matrix_t* normalize_action(matrix_t* action) {
  matrix_t* ret = matrix_clone(action);
  add_scalar(ret, act_clip_range);
  mult_scalar(ret, 1.0/(act_clip_range*2));
  mult_scalar(ret, 2);
  add_scalar(ret, 1);
  return ret;
}

#endif
