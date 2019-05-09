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
    if (fwrite(nxt_linear.W->data, sizeof(double), w_info[0]*w_info[1], fp) != w_info[0]*w_info[1]) {
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
    if (fwrite(nxt_linear.b->data, sizeof(double), b_rows*b_cols, fp) != b_rows*b_cols) {
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
    if (fread(m->hidden_linears[i].data.l.W->data, sizeof(double), w_rows*w_cols, fp) != w_rows*w_cols) {
      printf("[LOAD_MODEL] Failed to read layer weights, file could be corrupted, aborting...\n");
      return 0;
    }
    free(w_info);
    int b_rows = m->hidden_linears[i].data.l.b->rows;
    int b_cols = m->hidden_linears[i].data.l.b->cols;
    if (fread(m->hidden_linears[i].data.l.b->data, sizeof(double), b_rows*b_cols, fp) != b_rows*b_cols) {
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

