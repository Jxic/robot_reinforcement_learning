#include "utils.h"
#include <time.h>
#include <stdlib.h>
#include <assert.h>

double rand_uniform(double low, double high) {
  assert(high >= low);
  double range = high - low;
  return low + ((double)rand() / (double)RAND_MAX * range);
}

