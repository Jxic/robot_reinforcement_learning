#include "utils.h"
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "matrix_op.h"

#include <unistd.h>
#include <stdio.h>
#include <limits.h>

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
  double array[BUFFER_SIZE];
  double* w = array;
  remove_char(c, '\n');
  char* token = strtok(c, " ");
  int count = 0;
  while (token) {
    *w++ = convert_double(token);
    token = strtok(NULL, " ");
    count++;
  }
  if (create) {
    augment_space(m, 1, count);
    m->cols = count;
    memcpy(m->data, array, count*sizeof(double));
  } else {
    assert(count == m->cols);
    augment_space(m, m->rows+1, count);
    memcpy(m->data+(m->rows*m->cols), array, count*sizeof(double));
    m->rows++;
  }
  return 1;
}

static double convert_double(char* c) {
  if (c == NULL || *c == '\0' || isspace(*c)) {
    printf("[CONVERT_DOUBLE] failed to convert %s", c);
    exit(1);
  }
  char* p;
  double ret = strtod(c, &p);
  if ( *p != '\0') {
    printf("[CONVERT_DOUBLE] failed to convert %s", c);
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
