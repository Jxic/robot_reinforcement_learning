#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H
#include "model.h"
#include "normalizer.h"


int save_model(model* m, char* model_name);
model* load_model(char* model_name);
int save_normalizer(normalizer* n, char* n_name);
normalizer* load_normalizer(char* n_name);

matrix_t* flatten(matrix_t** ms, int count);
matrix_t** rebuild(model* m, matrix_t* flattened);

#endif
