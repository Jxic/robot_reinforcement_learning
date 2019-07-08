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
static double convert_double(char* c);
static int remove_char(char* s, char c);

double rand_uniform(double low, double high) {
  assert(high >= low);
  double range = high - low;
  return low + ((double)rand() / (double)RAND_MAX * range);
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
      perror("getcwd() error\n");
    }
    exit(1);
  }
  // int count = 1;
  //construct a matrix
  FILE* line_counter = fopen(filename, "r");
  int line_num = 0;
  while (fgets(buff, BUFFER_SIZE, line_counter)) {
    line_num++;
  }
  fclose(line_counter);
  matrix_t* new_mat = new_matrix(1, 1);
  if (fgets(buff, BUFFER_SIZE, fp)) {
    // printf("line %d\n", count++);
    append(new_mat, buff, line_num);
  }
  // printf("entering while\n");
  while (fgets(buff, BUFFER_SIZE, fp)) {
    // printf("line %d\n", count++);
    append(new_mat, buff, 0);
    // if (count > 100) {
    //   break;
    // }
  }
  fclose(fp);

  return new_mat;
}

static int append(matrix_t* m, char* c, int create) {
  // printf("appending %s\n", c);
  double array[BUFFER_SIZE];
  // double* array = calloc(BUFFER_SIZE, sizeof(double));
  double* w = array;
  remove_char(c, '\n');
  remove_char(c, '\r');
  char* token = strtok(c, " ,");
  int count = 0;
  // char last_char[64];
  while (token) {
    // printf("count %d token %c\n", count, *token);
    // if (!strcmp(token, last_char)) {
      // *w = *(w-1);
      // w++;
    // } else {
      *w++ = convert_double(token);
    //}
    // strcpy(last_char, token);
    token = strtok(NULL, " ,");
    count++;
    // printf("done\n");
  }
  // printf("converted\n");
  if (create) {
    augment_space(m, create, count);
    m->cols = count;
    memcpy(m->data, array, count*sizeof(double));
  } else {
    assert(count == m->cols);
    // augment_space(m, m->rows+1, count);
    memcpy(m->data+(m->rows*m->cols), array, count*sizeof(double));
    m->rows++;
  }
  // printf("returning\n");
  return 1;
}

static double convert_double(char* c) {
  // printf("converting raw %s\n",c);
  if (c == NULL || *c == '\0' || isspace(*c)) {
    printf("[CONVERT_DOUBLE] failed to convert %s\n", c);
    exit(1);
  }
  char* p;
  double ret = strtod(c, &p);
  // printf("converted to %f checking result\n",ret);
  if ( *p != '\0') {
    printf("[CONVERT_DOUBLE] failed to convert %s\n", c);
    printf("Remaining part %d\n", *p);
    exit(1);
  }
  // printf("returning converted double\n");
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
  double* values = v->data;
  //double average, deviation;
  if ( values ) {
    for ( i = 0; i < m; i += 2 ) {
      double x,y,rsq,f;
      do {
        x = 2.0 * rand() / (double)RAND_MAX - 1.0;
        y = 2.0 * rand() / (double)RAND_MAX - 1.0;
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

matrix_t* trunc_normal(int n, double high, double low) {
  int i = 0;
  int m = n + n % 2;
  matrix_t* v = new_matrix(1, m);
  double* values = v->data;
  //double average, deviation;
  if ( values ) {
    while (i < m) {
      double x,y,rsq,f;
      do {
        x = 2.0 * rand() / (double)RAND_MAX - 1.0;
        y = 2.0 * rand() / (double)RAND_MAX - 1.0;
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

double timer_check(struct timeval* t) {
  struct timeval end;
  gettimeofday(&end, NULL); 
  double time_taken;
  time_taken = (end.tv_sec - t->tv_sec) * 1e6;
  time_taken = (time_taken + (end.tv_usec - t->tv_usec)) * 1e-3; 
  timer_reset(t);
  return time_taken;
} 

double timer_observe(struct timeval* t) {
  struct timeval end;
  gettimeofday(&end, NULL); 
  double time_taken;
  time_taken = (end.tv_sec - t->tv_sec) * 1e6;
  time_taken = (time_taken + (end.tv_usec - t->tv_usec)) * 1e-3; 
  return time_taken;
} 

