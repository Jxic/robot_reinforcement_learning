#include "rl_utils.h"
#include <stdlib.h>
#include <assert.h>
#include "utils.h"
#include <string.h>
#include <stdio.h>
#include "socket_utils.h"
#include "sim_api.h"
#include <math.h>
#include "macros.h"
#ifdef MPI
#include "mpi.h"
#endif
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
  #ifdef MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  init_demo_connection(5555+rank);
  #else
  init_demo_connection(5555);
  #endif
  return build_demo_buffer(size, transition_dim);
  #else
  printf("[INIT_DEMO_BUFFER] C_AS_LIB\n");
  return build_sim_demo_buffer(size, transition_dim);
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
    memcpy(ret->data+(i*ret->cols), exp_buf->experiences[idx]->data, ret->cols*sizeof(float));
  }

  return ret;
}

void print_experiences(experience_buffer* exp_buf) {
  assert(exp_buf->curr_size > 0);
  matrix_t* all = new_matrix(exp_buf->curr_size, exp_buf->experiences[0]->cols);
  for (int i = 0; i < exp_buf->curr_size; ++i) {
    memcpy(all->data+(i*all->cols), exp_buf->experiences[i]->data, all->cols*sizeof(float));
  }
  print_matrix(all, 1);
}

#ifdef C_AS_LIB

static matrix_t** go_to_point(matrix_t* obs, matrix_t* pos, int* timestep);
static matrix_t** fill_up_episode(matrix_t* obs, int curr_time_step, int to_time_step);
static float distance(matrix_t* a, matrix_t* b);
static matrix_t* normalize_action(matrix_t* action);

static int reward_dim = 1;
static int info_dim = 1;
static int act_dim = 4;
static int g_dim = 3;
static int obj_pos_offset = 7;
static int state_dim = 14;
static float dist_threshold = 0.05;

static int rand_angle = 0;
static int rand_dest_pos = 1;
static float act_clip_range = 0.0872665; // 5 degrees
static int env_step_limit = 50;

static experience_buffer* build_sim_demo_buffer(int size, int transition_dim) {
  assert(transition_dim == 2*state_dim + act_dim + info_dim + reward_dim);
  printf("Initializing demo buffer to size: %d\n", size);
  experience_buffer* ret = init_experience_buffer(size*env_step_limit);
  
  printf("Building buffer with state dimension: %d, action dimension: %d, total transition dimension: %d\n", state_dim, act_dim, transition_dim);

  initEnv(0, ENV_PICK_N_PLACE);

  for (int i = 0; i < size; ++i) {
    matrix_t* init_obs = resetState(rand_angle, rand_dest_pos, 0, 0);
    print_matrix(init_obs, 1);
    int time_step = 0;
    // Reach the target with end effector
    matrix_t* first_stop = slice_col_wise(init_obs, obj_pos_offset, obj_pos_offset+g_dim);
    matrix_t** transitions = go_to_point(init_obs, first_stop, &time_step);
    if (time_step > env_step_limit) {
      printf("Too many steps taken to reach the target, resampling target\n");
      i--;
      exit(1);
    }
    for (int j = 0; j < time_step; ++j) {
      assert(transition_dim == transitions[j]->cols);
      store_experience(ret, transitions[j]);
    }
    printf("%d Reached target\n", i);
    // exit(1);
    // Grab the object
    matrix_t* last_transition = transitions[time_step-1];
    matrix_t* last_obs = slice_col_wise(last_transition, state_dim+act_dim, state_dim+act_dim+state_dim);
    matrix_t* grab_act = new_matrix(1, act_dim);
    initialize(grab_act, zeros);
    grab_act->data[grab_act->cols-1] = 1;
    matrix_t* nxt_state = step(grab_act, 0, 0);
    //print_matrix(nxt_state, 1);
    matrix_t* nxt_obs = slice_col_wise(nxt_state, 0, state_dim);
    matrix_t* other_info = slice_col_wise(nxt_state, state_dim+g_dim, nxt_state->cols);
    matrix_t* o_a = concatenate(last_obs, grab_act, 1);
    matrix_t* o_a_o2 = concatenate(o_a, nxt_obs, 1);
    matrix_t* o_a_o2_dr = concatenate(o_a_o2, other_info, 1);
    time_step++;
    if (time_step > env_step_limit) {
      printf("Too many steps taken to grab the target, resampling target\n");
      i--;
      exit(1);
    }
    printf("%d Grabbed target\n", i);
    assert(transition_dim == o_a_o2_dr->cols);
    store_experience(ret, o_a_o2_dr);

    // Deliver the target
    
    matrix_t* nxt_stop = slice_col_wise(nxt_state, state_dim-g_dim, state_dim);
    matrix_t* inter_stop = matrix_clone(first_stop);
    elem_wise_add(inter_stop, nxt_stop);
    mult_scalar(inter_stop, 0.5);
    // printf("%d inter stop\n", i);
    inter_stop->data[1] = distance(first_stop, nxt_stop) / 3;
    // printf("%d calculated distance\n", i);
    matrix_t* fst_quater_stop = matrix_clone(first_stop);
    matrix_t* snd_quater_stop = matrix_clone(nxt_stop);
    // elem_wise_add(fst_quater_stop, first_stop);
    // mult_scalar(fst_quater_stop, 0.5);
    // elem_wise_add(snd_quater_stop, nxt_stop);
    // mult_scalar(snd_quater_stop, 0.5);
    fst_quater_stop->data[1] = snd_quater_stop->data[1] = inter_stop->data[1];
    // printf("%d got quater stops\n", i);

    int fst_quater_start_time_step = time_step;
    matrix_t** fst_quater_transition = go_to_point(nxt_state, fst_quater_stop, &time_step);
    matrix_t* half_lifted_state = fst_quater_start_time_step != time_step ? \
      slice_col_wise(fst_quater_transition[time_step-1], state_dim+act_dim, state_dim+act_dim+state_dim) : nxt_state;
    printf("%d reached first quater\n", i);
    int lift_start_time_step = time_step;
    matrix_t** inter_transition = go_to_point(half_lifted_state, inter_stop, &time_step);
    matrix_t* lifted_state = lift_start_time_step != time_step ? \
      slice_col_wise(inter_transition[time_step-1], state_dim+act_dim, state_dim+act_dim+state_dim) : half_lifted_state; 
    printf("%d reached inter\n", i);
    int snd_quater_start_time_step = time_step;
    matrix_t** snd_quater_transition = go_to_point(lifted_state, snd_quater_stop, &time_step);
    matrix_t* half_descend_state = snd_quater_start_time_step != time_step ? \
     slice_col_wise(snd_quater_transition[time_step-1], state_dim+act_dim, state_dim+act_dim+state_dim) : lifted_state;
    printf("%d reached second quater\n", i);
    int deliver_start_time_step = time_step;
    matrix_t** phase_2_transition = go_to_point(half_descend_state ,nxt_stop, &time_step);
    if (time_step > env_step_limit) {
      printf("Too many steps taken to deliver the target, resampling target\n");
      i--;
      exit(1);
    }
    for (int j = fst_quater_start_time_step; j < lift_start_time_step; ++j) {
      // printf("transition at: %d\n", j);
      assert(transition_dim == fst_quater_transition[j]->cols);
      store_experience(ret, fst_quater_transition[j]);
    }
    for (int j = lift_start_time_step; j < snd_quater_start_time_step; ++j) {
      // printf("transition at: %d\n", j);
      assert(transition_dim == inter_transition[j]->cols);
      store_experience(ret, inter_transition[j]);
    }
    for (int j = snd_quater_start_time_step; j < deliver_start_time_step; ++j) {
      // printf("transition at: %d\n", j);
      assert(transition_dim == snd_quater_transition[j]->cols);
      store_experience(ret, snd_quater_transition[j]);
    }
    for (int j = deliver_start_time_step; j < time_step; ++j) {
      // printf("transition at: %d\n", j);
      assert(transition_dim == phase_2_transition[j]->cols);
      store_experience(ret, phase_2_transition[j]);
    }
    printf("%d Delivered target\n", i);
    // Fill up the episode with padding
    matrix_t* phase_2_last_transition = phase_2_transition[time_step-1];
    matrix_t* phase_2_last_obs = slice_col_wise(phase_2_last_transition, state_dim+act_dim, phase_2_last_transition->cols);
    // printf("%d Actually delivered target: %f\n", i, phase_2_last_obs->data[state_dim-g_dim-1]);
    assert(phase_2_last_obs->data[state_dim-g_dim-1]);
    matrix_t** padding_transitions = fill_up_episode(phase_2_last_obs, time_step, env_step_limit);
    for (int j = time_step; j < env_step_limit; ++j) {
      assert(transition_dim == padding_transitions[j]->cols);
      store_experience(ret, padding_transitions[j]);
    }
    printf("%d Filled up episode\n", i);

    // printf("experience size: %d\n", ret->curr_size);
    // check behaviour with rendering
    matrix_t** all_actions = calloc(env_step_limit, sizeof(*all_actions));
    for (int j = 0; j < env_step_limit; ++j) {
      int offset = i * env_step_limit;
      // printf("act: %d exp curr size: %d\n", i, ret->curr_size);
      // print_matrix(ret->experiences[j+offset], 1);
      matrix_t* nxt_act = slice_col_wise(ret->experiences[j+offset], state_dim, state_dim+act_dim);
      all_actions[j] = nxt_act;
      // free resources

    }
    
    #ifdef RENDER
    printf("Ready for rendering\n");
    renderSteps(all_actions, env_step_limit);
    #endif

    // free resources
    for (int j = 0; j < env_step_limit; ++j) {
      free_matrix(all_actions[j]);
    }
    free(inter_transition);
    free_matrix(lifted_state);
    free_matrix(inter_stop);
    free(all_actions);
    free(padding_transitions);
    free_matrix(phase_2_last_obs);
    free(phase_2_transition);
    free_matrix(nxt_stop);
    free_matrix(o_a_o2);
    free_matrix(o_a);
    free_matrix(other_info);
    free_matrix(nxt_obs);
    free_matrix(nxt_state);
    free_matrix(grab_act);
    free_matrix(last_obs);
    free(transitions);
    free_matrix(first_stop);
    free_matrix(init_obs);


  }

  return ret;
}


static matrix_t** go_to_point(matrix_t* obs, matrix_t* pos, int* timestep) {
  //observation: joint angles(3), ee position(3), ee state(1), object position(3), has object(1)
  //target position (dest position): (3)
  matrix_t* state = slice_col_wise(obs, 0, state_dim);
  matrix_t* state_ee_pos = slice_col_wise(state, g_dim, 2*g_dim);
  // printf("going from: \n");
  // print_matrix(state_ee_pos, 1);
  // printf("to :\n");
  // print_matrix(pos, 1);
  matrix_t** ret = calloc(env_step_limit, sizeof(*ret));
  // printf("Distance to go %f\n", distance(state_ee_pos, pos));
  while(distance(state_ee_pos, pos) > dist_threshold) {
    
    matrix_t* nxt_action = inverse_km(pos);

    matrix_t* curr_ja = slice_col_wise(state, 0, g_dim);
    elem_wise_minus(nxt_action, curr_ja);

    clip(nxt_action, -act_clip_range, act_clip_range);
    matrix_t* normed_action = normalize_action(nxt_action);
    augment_space(normed_action, 1, act_dim);
    normed_action->cols = act_dim;
    normed_action->data[act_dim-1] = state->data[g_dim*2];

    clip(normed_action, -1, 1);
    matrix_t* nxt_obs = step(normed_action, 0, 0);
    // printf("nxt: %d\n", *timestep);
    // print_matrix(nxt_obs, 1);
    matrix_t* o2 = slice_col_wise(nxt_obs, 0, state_dim);
    matrix_t* dr = slice_col_wise(nxt_obs, state_dim+g_dim, nxt_obs->cols);
    
    matrix_t* o_a = concatenate(state, normed_action, 1);
    matrix_t* o_a_o2 = concatenate(o_a, o2, 1);
    matrix_t* o_a_o2_dr = concatenate(o_a_o2, dr, 1);

    free_matrix(state);
    free_matrix(state_ee_pos);
    state = slice_col_wise(nxt_obs, 0, state_dim);
    state_ee_pos = slice_col_wise(state, g_dim, 2*g_dim);

    ret[*timestep] = o_a_o2_dr;
    *timestep += 1;

    free_matrix(o_a_o2);
    free_matrix(o_a);
    free_matrix(dr);
    free_matrix(o2);
    free_matrix(nxt_obs);
    free_matrix(normed_action);
    free_matrix(curr_ja);
    free_matrix(nxt_action);

  }
  return ret;
}

static matrix_t** fill_up_episode(matrix_t* obs, int curr_time_step, int to_time_step) {
  matrix_t** ret = calloc(to_time_step, sizeof(*ret));
  matrix_t* state = slice_col_wise(obs, 0, state_dim);
  matrix_t* action = new_matrix(1, act_dim);
  initialize(action, zeros);
  // action->data[action->cols-1] = 1;
  for (int i = curr_time_step; i < to_time_step; ++i) {
    matrix_t* o_a = concatenate(state, action, 1);
    matrix_t* o_a_o2_dr = concatenate(o_a, obs, 1);
    ret[i] = o_a_o2_dr;
    free_matrix(o_a);
  }
  return ret;
}

static float distance(matrix_t* a, matrix_t* b) {
  assert(a->rows == b->rows);
  assert(a->rows == 1);
  matrix_t* temp = matrix_clone(a);
  elem_wise_minus(temp, b);
  elem_wise_mult(temp, temp);
  float sum = 0;
  for (int i = 0; i < a->cols; ++i) {
    sum += temp->data[i];
  }
  free_matrix(temp);
  return sqrt(sum);
}

static matrix_t* normalize_action(matrix_t* action) {
  matrix_t* ret = matrix_clone(action);
  add_scalar(ret, act_clip_range);
  mult_scalar(ret, 1.0/(act_clip_range*2));
  mult_scalar(ret, 2);
  add_scalar(ret, -1);
  return ret;
}

#endif
