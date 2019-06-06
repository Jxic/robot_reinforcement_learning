#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#ifdef GPU
#include "layers.h"
#include "matrix_op.h"

#define SGD_LR 0.01
#define ADAM_LR 0.001

typedef struct _adam_optimizer {
  int timestamp;
  double beta1;
  double beta2;
  // layer* first_moment;
  // layer* second_moment;
  matrix_t* first_moment;
  matrix_t* second_moment;
  double** trainable_params;
  double** trainable_params_g;
  double epsilon;
  double learning_rate;
  int num_of_layers;
} adam_optimizer;

typedef struct _sgd_optimizer {
  double learning_rate;
} sgd_optimizer;

typedef enum _optimizer_type {
  sgd, adam, no_opt
} optimizer_type;

typedef union _optimizer_cache {
  adam_optimizer a;
  sgd_optimizer s;
} optimizer_cache;

typedef struct _optimizer {
  optimizer_type type;
  optimizer_cache cache;
} optimizer;



int free_optimizer(optimizer o);
#else

#include "layers.h"

#define SGD_LR 0.01
#define ADAM_LR 0.001

typedef struct _adam_optimizer {
  int timestamp;
  double beta1;
  double beta2;
  layer* first_moment;
  layer* second_moment;
  double epsilon;
  double learning_rate;
  int num_of_layers;
} adam_optimizer;

typedef struct _sgd_optimizer {
  double learning_rate;
} sgd_optimizer;

typedef enum _optimizer_type {
  sgd, adam, no_opt
} optimizer_type;

typedef union _optimizer_cache {
  adam_optimizer a;
  sgd_optimizer s;
} optimizer_cache;

typedef struct _optimizer {
  optimizer_type type;
  optimizer_cache cache;
} optimizer;



int free_optimizer(optimizer o);


#endif
#endif


