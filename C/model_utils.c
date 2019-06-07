#include <assert.h>
#include "model_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "layers.h"
#include <string.h>
#include "optimizer.h"

#define DEFAULT_MODEL_DIR "./models/"
#define DEFAULT_MODE 0744
#define MODEL_META_INFO_NUM 7
#define PATH_BUFFER_SIZE 1024
#define NORMALIZER_META_INFO_NUM 2

int save_model(model* m, char* model_name) {
  printf("Saving data ... \n");
  FILE* fp;
  struct stat s;
  // create sub-directory if not exist
  if (stat(DEFAULT_MODEL_DIR, &s) == -1) {
    mkdir(DEFAULT_MODEL_DIR, DEFAULT_MODE);
  }
  char path[PATH_BUFFER_SIZE];
  strcpy(path, DEFAULT_MODEL_DIR);
  strcat(path, model_name);
  // open file and prepare for write
  fp = fopen(path, "w+");
  int* model_meta = calloc(MODEL_META_INFO_NUM, sizeof(int));
  model_meta[0] = m->input_dim;
  model_meta[1] = m->output_dim;
  model_meta[2] = m->num_of_layers;
  model_meta[3] = m->max_out;
  model_meta[4] = m->loss_layer.type;
  model_meta[5] = m->version;
  model_meta[6] = m->optimizer.type;

  // write meta info to file and iteratively save layers' weights
  if (fwrite(model_meta, sizeof(int), MODEL_META_INFO_NUM, fp) != MODEL_META_INFO_NUM) {
    printf("[SAVE_MODEL] Failed to write model meta information to file, aborting...\n");
    return 0;
  }
  free(model_meta);

  for(int i = 0; i < m->num_of_layers; ++i) {
    linear_layer nxt_linear = m->hidden_linears[i].data.l;
    int* w_info = calloc(2, sizeof(int));
    w_info[0] = nxt_linear.W->rows;
    w_info[1] = nxt_linear.W->cols;
    if (fwrite(w_info, sizeof(int), 2, fp) != 2) {
      printf("[SAVE_MODEL] Failed to write layer weights meta information to file, aborting...\n");
      return 0;
    }
    layer_type act_type = m->hidden_activations[i].type;
    if (fwrite(&act_type, sizeof(layer_type), 1, fp) != 1) {
      printf("[SAVE_MODEL] Failed to write layer activation to file, aborting...\n");
      return 0;
    }
    if (fwrite(nxt_linear.W->data, sizeof(float), w_info[0]*w_info[1], fp) != w_info[0]*w_info[1]) {
      printf("[SAVE_MODEL] Failed to write layer weights to file, aborting...\n");
      return 0;
    }
    free(w_info);
    int b_rows = nxt_linear.b->rows;
    int b_cols = nxt_linear.b->cols;
    // if (fwrite(b_info, sizeof(int), 2, fp) != 2) {
    //   printf("[SAVE_MODEL] Failed to write layer bias meta information to file, aborting...\n");
    //   return 0;
    // }
    if (fwrite(nxt_linear.b->data, sizeof(float), b_rows*b_cols, fp) != b_rows*b_cols) {
      printf("[SAVE_MODEL] Failed to write layer bias to file, aborting...\n");
      return 0;
    }
  }
  fclose(fp);
  printf("Model saved to %s\n", path);
  return 1;
}

model* load_model(char* model_name) {
  printf("Loading model ... \n");
  FILE* fp;
  struct stat s;
  // look for default directory
  if (stat(DEFAULT_MODEL_DIR, &s) == -1) {
    printf("[LOAD_MODEL] model directory not found, no model has been created before\n");
    return 0;
  }
  // look for model name
  char path[PATH_BUFFER_SIZE];
  strcpy(path, DEFAULT_MODEL_DIR);
  strcat(path, model_name);
  fp = fopen(path, "r");
  if (!fp) {
    printf("[LOAD_DATA] file not existed, %s\n", path);
    char cwd[100];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
      printf("Current working dir: %s\n", cwd);
    } else {
      perror("getcwd() error");
    }
    exit(1);
  }
  // reconstructing model
  int* model_meta = calloc(MODEL_META_INFO_NUM, sizeof(int));
  if (fread(model_meta, sizeof(int), MODEL_META_INFO_NUM, fp) != MODEL_META_INFO_NUM) {
    printf("[LOAD_MODEL] Failed to read model meta information, file could be corrupted, aborting...\n");
    return 0;
  }

  int input_dim = model_meta[0];
  int output_dim = model_meta[1];
  int num_of_layers = model_meta[2];
  int max_out = model_meta[3];
  layer_type loss_type = model_meta[4];
  int version = model_meta[5];
  optimizer_type opt_type = model_meta[6];

  model* m = init_model(input_dim);
  //m->output_dim = output_dim;
  //m->num_of_layers = num_of_layers;
  //m->max_out = max_out;
  m->version = version;
  m->cache_initialized = 0;

  for (int i = 0; i < num_of_layers; ++i) {
    int* w_info = calloc(2, sizeof(int));
    if (fread(w_info, sizeof(int), 2, fp) != 2) {
      printf("[LOAD_MODEL] Failed to read layer weights meta information, file could be corrupted, aborting...\n");
      return 0;
    }
    int w_rows = w_info[0];
    int w_cols = w_info[1];
    layer_type act_type;
    if (fread(&act_type, sizeof(layer_type), 1, fp) != 1) {
      printf("[LOAD_MODEL] Failed to read layer activation information, file could be corrupted, aborting...\n");
      return 0;
    }
    add_linear_layer(m, w_cols, act_type);
    if (fread(m->hidden_linears[i].data.l.W->data, sizeof(float), w_rows*w_cols, fp) != w_rows*w_cols) {
      printf("[LOAD_MODEL] Failed to read layer weights, file could be corrupted, aborting...\n");
      return 0;
    }
    free(w_info);
    int b_rows = m->hidden_linears[i].data.l.b->rows;
    int b_cols = m->hidden_linears[i].data.l.b->cols;
    if (fread(m->hidden_linears[i].data.l.b->data, sizeof(float), b_rows*b_cols, fp) != b_rows*b_cols) {
      printf("[LOAD_MODEL] Failed to read layer bias, file could be corrupted, aborting...\n");
      return 0;
    }
  }

  compile_model(m, loss_type, opt_type);
  assert(m->output_dim == output_dim);
  assert(m->max_out == max_out);
  assert(m->num_of_layers == num_of_layers);
  fclose(fp);
  free(model_meta);
  printf("Model loaded from %s, remember to reinitialize cache for retraining or evaluation ...\n", path);
  return m;
}


int save_normalizer(normalizer* m, char* n_name) {
  printf("Saving normalizer ... \n");
  FILE* fp;
  struct stat s;
  // create sub-directory if not exist
  if (stat(DEFAULT_MODEL_DIR, &s) == -1) {
    mkdir(DEFAULT_MODEL_DIR, DEFAULT_MODE);
  }
  char path[PATH_BUFFER_SIZE];
  strcpy(path, DEFAULT_MODEL_DIR);
  strcat(path, n_name);
  // open file and prepare for write
  fp = fopen(path, "w+");
  float* n_meta = calloc(NORMALIZER_META_INFO_NUM, sizeof(float));
  n_meta[0] = (float)m->n;
  n_meta[1] = m->clip_value;

  // write meta info to file and iteratively save layers' weights
  if (fwrite(n_meta, sizeof(float), NORMALIZER_META_INFO_NUM, fp) != NORMALIZER_META_INFO_NUM) {
    printf("[SAVE_NORMALIZER] Failed to write NORMALIZER meta information to file, aborting...\n");
    return 0;
  }
  free(n_meta);

  matrix_t* mean = m->mean;
  int* w_info = calloc(2, sizeof(int));
  w_info[0] = mean->rows;
  w_info[1] = mean->cols;
  if (fwrite(w_info, sizeof(int), 2, fp) != 2) {
    printf("[SAVE_NORMALIZER] Failed to write mean meta information to file, aborting...\n");
    return 0;
  }
  if (fwrite(mean->data, sizeof(float), w_info[0]*w_info[1], fp) != w_info[0]*w_info[1]) {
    printf("[SAVE_NORMALIZER] Failed to write mean to file, aborting...\n");
    return 0;
  }
  free(w_info);
  matrix_t* sum = m->sum;
  int b_rows = sum->rows;
  int b_cols = sum->cols;
  if (fwrite(sum->data, sizeof(float), b_rows*b_cols, fp) != b_rows*b_cols) {
    printf("[SAVE_NORMALIZER] Failed to write layer bias to file, aborting...\n");
    return 0;
  }
  matrix_t* std = m->std;
  int v_rows = std->rows;
  int v_cols = std->cols;
  if (fwrite(std->data, sizeof(float), v_rows*v_cols, fp) != v_rows*v_cols) {
    printf("[SAVE_NORMALIZER] Failed to write layer bias to file, aborting...\n");
    return 0;
  }
  matrix_t* sumsq = m->sumsq;
  if (fwrite(sumsq->data, sizeof(float), v_rows*v_cols, fp) != v_rows*v_cols) {
    printf("[SAVE_NORMALIZER] Failed to write layer bias to file, aborting...\n");
    return 0;
  }
  fclose(fp);
  printf("NORMALIZER saved to %s\n", path);
  return 1;
}

normalizer* load_normalizer(char* n_name) {
  printf("Loading normalizer ... \n");
  FILE* fp;
  struct stat s;
  // look for default directory
  if (stat(DEFAULT_MODEL_DIR, &s) == -1) {
    printf("[LOAD_NORMALIZER] NORMALIZER directory not found, no NORMALIZER has been created before\n");
    return 0;
  }
  // look for model name
  char path[PATH_BUFFER_SIZE];
  strcpy(path, DEFAULT_MODEL_DIR);
  strcat(path, n_name);
  fp = fopen(path, "r");
  if (!fp) {
    printf("[LOAD_DATA] file not existed, %s\n", path);
    char cwd[100];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
      printf("Current working dir: %s\n", cwd);
    } else {
      perror("getcwd() error");
    }
    exit(1);
  }
  // reconstructing normalizer
  float* n_meta = calloc(NORMALIZER_META_INFO_NUM, sizeof(float));
  if (fread(n_meta, sizeof(float), NORMALIZER_META_INFO_NUM, fp) != NORMALIZER_META_INFO_NUM) {
    printf("[LOAD_NORMALIZER] Failed to read NORMALIZER meta information, file could be corrupted, aborting...\n");
    return 0;
  }

  int n = (int)n_meta[0];
  float clip_value = n_meta[1];

  int* w_info = calloc(2, sizeof(int));
  if (fread(w_info, sizeof(int), 2, fp) != 2) {
    printf("[LOAD_NORMALIZER] Failed to read mean meta information, file could be corrupted, aborting...\n");
    return 0;
  }
  int w_rows = w_info[0];
  int w_cols = w_info[1];
  
  normalizer* norm = init_normalizer(w_cols, clip_value);
  norm->n = n;

  if (fread(norm->mean->data, sizeof(float), w_rows*w_cols, fp) != w_rows*w_cols) {
    printf("[LOAD_NORMALIZER] Failed to read means, file could be corrupted, aborting...\n");
    return 0;
  }

  if (fread(norm->sum->data, sizeof(float), w_rows*w_cols, fp) != w_rows*w_cols) {
    printf("[LOAD_NORMALIZER] Failed to read sums, file could be corrupted, aborting...\n");
    return 0;
  }

  if (fread(norm->std->data, sizeof(float), w_rows*w_cols, fp) != w_rows*w_cols) {
    printf("[LOAD_NORMALIZER] Failed to read stds, file could be corrupted, aborting...\n");
    return 0;
  }

  if (fread(norm->sumsq->data, sizeof(float), w_rows*w_cols, fp) != w_rows*w_cols) {
    printf("[LOAD_NORMALIZER] Failed to read sumsqs, file could be corrupted, aborting...\n");
    return 0;
  }
  free(w_info);
  fclose(fp);
  free(n_meta);
  printf("Normalizer loaded from %s ...\n", path);
  return norm;
}


matrix_t* flatten(matrix_t** ms, int count) {
  assert(count > 0);
  int size = 0;
  for (int i = 0; i < count; ++i) {
    size += ms[i]->cols*ms[i]->rows;
  }

  int start = 0;
  matrix_t* ret = new_matrix(1, size);
  for (int i = 0; i < count; ++i) {
    memcpy(ret->data+start, ms[i]->data, ms[i]->rows*ms[i]->cols*sizeof(float));
    start += ms[i]->rows*ms[i]->cols;
  }

  return ret;
}

matrix_t** rebuild(model* m, matrix_t* flattened) {
  assert(flattened->rows == 1);
  assert(flattened->cols > 0);

  matrix_t** ret = calloc(2*m->num_of_layers, sizeof(*ret));
  int start = 0;
  for (int i = 0; i < 2*m->num_of_layers; i+=2) {
    int w_rows = m->hidden_linears[i/2].data.l.W->rows;
    int w_cols = m->hidden_linears[i/2].data.l.W->cols;
    int b_rows = m->hidden_linears[i/2].data.l.b->rows;
    int b_cols = m->hidden_linears[i/2].data.l.b->cols;
    matrix_t* W = new_matrix(w_rows, w_cols);
    matrix_t* b = new_matrix(b_rows, b_cols);
    assert(start < flattened->cols);
    memcpy(W->data, flattened->data+start, w_rows*w_cols*sizeof(float));
    start += w_rows*w_cols;
    assert(start < flattened->cols);
    memcpy(b->data, flattened->data+start, b_rows*b_cols*sizeof(float));
    start += b_rows*b_cols;
    ret[i] = W;
    ret[i+1] = b;
  }
  
  return ret;
}

