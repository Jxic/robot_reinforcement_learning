#ifndef NORMALIZER_H
#define NORMALIZER_H

#include "matrix_op.h"

#define DEFAULT_CLIP_RANGE 5

typedef struct _normalizer {
  int n;
  matrix_t* mean;
  matrix_t* mean_diff;
  matrix_t* var;
  double clip_value;
} normalizer;

normalizer* init_normalizer(int dim, double clip_range);
int normalize_obs(normalizer* n, matrix_t* states);
int update_normalizer(normalizer* n, matrix_t** observations, int count);


#endif
