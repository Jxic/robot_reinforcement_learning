#ifndef MPI_UTILS_H
#define MPI_UTILS_H
#include "../macros.h"
#ifdef MPI
#include "../model.h"
#include "../normalizer.h"
#include "../model_utils.h"
#include "mpi.h"

int mpi_init();
int mpi_perform_update(model* m, double lr, int scale);
int mpi_update_normalizer(normalizer* norm, matrix_t** data, int count);
int mpi_sync(model* m);
int mpi_check_sync(model* m);
int mpi_finalize();

#endif
#endif
