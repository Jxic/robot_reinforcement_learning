#include "utils.h"
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "matrix_op.h"
#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <limits.h>
#include "macros.h"
#include <sys/time.h>
#ifdef MKL
#include "mkl.h"
#endif

static int append(matrix_t* m, char* c, int create);
static float convert_float(char* c);
static int remove_char(char* s, char c);

float rand_uniform(float low, float high) {
  assert(high >= low);
  float range = high - low;
  return low + ((float)rand() / (float)RAND_MAX * range);
}

matrix_t* load_data(char* filename) {
  // reading file
  FILE* fp;
  char buff[BUFFER_SIZE];
  fp = fopen(filename, "r");
  if (!fp) {
    printf("[LOAD_DATA] file not existed, %s\n", filename);
    char cwd[100];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
      printf("Current working dir: %s\n", cwd);
    } else {
      perror("getcwd() error");
    }
    exit(1);
  }
  //construct a matrix
  matrix_t* new_mat = new_matrix(1, 1);
  if (fgets(buff, BUFFER_SIZE, fp)) {
    append(new_mat, buff, 1);
  }
  while (fgets(buff, BUFFER_SIZE, fp)) {
    append(new_mat, buff, 0);
  }
  fclose(fp);

  return new_mat;
}

static int append(matrix_t* m, char* c, int create) {
  float array[BUFFER_SIZE];
  float* w = array;
  remove_char(c, '\n');
  char* token = strtok(c, " ");
  int count = 0;
  while (token) {
    *w++ = convert_float(token);
    token = strtok(NULL, " ");
    count++;
  }
  if (create) {
    augment_space(m, 1, count);
    m->cols = count;
    memcpy(m->data, array, count*sizeof(float));
  } else {
    assert(count == m->cols);
    augment_space(m, m->rows+1, count);
    memcpy(m->data+(m->rows*m->cols), array, count*sizeof(float));
    m->rows++;
  }
  return 1;
}

static float convert_float(char* c) {
  if (c == NULL || *c == '\0' || isspace(*c)) {
    printf("[CONVERT_float] failed to convert %s", c);
    exit(1);
  }
  char* p;
  float ret = strtod(c, &p);
  if ( *p != '\0') {
    printf("[CONVERT_float] failed to convert %s", c);
    exit(1);
  }
  return ret;
}

static int remove_char(char* s, char c) {
  char* r = s;
  char* w = s;
  while (*r) {
    *w = *r++;
    w += (*w != c);
  }
  *w = '\0';
  return 1;
}

matrix_t* rand_normal(int n) {
  int i;
  int m = n + n % 2;
  matrix_t* v = new_matrix(1, m);
  float* values = v->data;
  //float average, deviation;
  if ( values ) {
    for ( i = 0; i < m; i += 2 ) {
      float x,y,rsq,f;
      do {
        x = 2.0 * rand() / (float)RAND_MAX - 1.0;
        y = 2.0 * rand() / (float)RAND_MAX - 1.0;
        rsq = x * x + y * y;
      }while( rsq >= 1. || rsq == 0. );
      f = sqrt( -2.0 * log(rsq) / rsq );
      values[i]   = x * f;
      values[i+1] = y * f;
    }
  }
  matrix_t* ret = slice_col_wise(v, 0, n);
  free_matrix(v);
  return ret;
}

matrix_t* trunc_normal(int n, float high, float low) {
  int i = 0;
  int m = n + n % 2;
  matrix_t* v = new_matrix(1, m);
  float* values = v->data;
  //float average, deviation;
  if ( values ) {
    while (i < m) {
      float x,y,rsq,f;
      do {
        x = 2.0 * rand() / (float)RAND_MAX - 1.0;
        y = 2.0 * rand() / (float)RAND_MAX - 1.0;
        rsq = x * x + y * y;
      }while( rsq >= 1. || rsq == 0. );
      f = sqrt( -2.0 * log(rsq) / rsq );
      if (x*f>high || x*f<low || y*f>high || y*f<low) {
        continue;
      }
      values[i]   = x * f;
      values[i+1] = y * f;
      i += 2;
    }
  }
  matrix_t* ret = slice_col_wise(v, 0, n);
  free_matrix(v);
  return ret;
}

void timer_reset(struct timeval* t) {
  gettimeofday(t, NULL); 
}

float timer_check(struct timeval* t) {
  struct timeval end;
  gettimeofday(&end, NULL); 
  float time_taken;
  time_taken = (end.tv_sec - t->tv_sec) * 1e6;
  time_taken = (time_taken + (end.tv_usec - t->tv_usec)) * 1e-3; 
  timer_reset(t);
  return time_taken;
} 

float timer_observe(struct timeval* t) {
  struct timeval end;
  gettimeofday(&end, NULL); 
  float time_taken;
  time_taken = (end.tv_sec - t->tv_sec) * 1e6;
  time_taken = (time_taken + (end.tv_usec - t->tv_usec)) * 1e-3; 
  return time_taken;
} 

