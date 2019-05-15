#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H
#include "model.h"
#include "normalizer.h"

int save_model(model* m, char* model_name);
model* load_model(char* model_name);
int save_normalizer(normalizer* n, char* n_name);
normalizer* load_normalizer(char* n_name);


#endif
