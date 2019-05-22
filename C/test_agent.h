#ifndef TEST_AGENT_H
#define TEST_AGENT_H
#include "matrix_op.h"

void run_agent(char* model_name, int with_normalizer, char* norm_name, int task_flag);
matrix_t** collect_trace(char* c, int w, char* n, int task_flag);

#endif
