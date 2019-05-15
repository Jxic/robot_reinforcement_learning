#include "test_agent.h"
#include "matrix_op.h"
#include "model.h"
#include "normalizer.h"
#include "model_utils.h"
#include "sim_api.h"

#define TEST_LOOP 50
#define TEST_RAND_ANGLE 0
#define TEST_RAND_DEST 1

static matrix_t* get_action(model* m, matrix_t* state);

void run_agent(char* model_name, int with_normalizer, char* norm_name) {
  model* actor;
  normalizer* norm = 0;
  actor = load_model(model_name);
  if (with_normalizer) {
    norm = load_normalizer(norm_name);
  }
  int state_dim = actor->input_dim;
  int act_dim = actor->output_dim;
  initEnv(act_dim);
  for (int k = 0; k < TEST_LOOP; ++k) {
    matrix_t* obs = resetState(TEST_RAND_ANGLE, TEST_RAND_DEST, state_dim+3, act_dim);
    obs = slice_col_wise(obs, 0, state_dim);
    if (with_normalizer) {
      normalize_obs(norm, obs);
    }
    for (int i = 0; i < TEST_LOOP; ++i) {
      matrix_t* action = get_action(actor, obs);
      matrix_t* nxt_state = step(action, state_dim+3, act_dim);
      free_matrix(obs);
      free_matrix(action);
      obs = nxt_state;
      obs = slice_col_wise(obs, 0, state_dim);
      if (with_normalizer) {
        normalize_obs(norm, obs);
      }
    }
  }
}

static matrix_t* get_action(model* m, matrix_t* state) {
  int state_dim = m->input_dim;
  int action_bound = 1;
  matrix_t* action = slice_col_wise(state, 0, state_dim);
  augment_space(action, action->rows, m->max_out);
  predict(m, action);
  mult_scalar(action, action_bound);
  clip(action, -action_bound, action_bound);
  return action;
}
