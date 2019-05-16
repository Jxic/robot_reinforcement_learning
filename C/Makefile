# for debugging checking memory leaks
DEBUG_CC = /usr/local/Cellar/llvm/8.0.0/bin/clang
CF_DEBUG_LAGS = -fsanitize=address -g

DEPS = matrix_op.h tests.h utils.h rl.h layers.h data_structures.h macros.h model.h sim_api.h rl_utils.h rl_ddpg.h socket_utils.h optimizer.h model_utils.h rl_ddpg_her.h normalizer.h test_agent.h
BDIR = OBJ_FILES
OBJ = main.o matrix_op.o tests.o utils.o rl.o layers.o model.o rl_utils.o rl_ddpg.o socket_utils.o optimizer.o model_utils.o sim_api.o rl_ddpg_her.o normalizer.o test_agent.c
EXE = main

# for production
CC = gcc
CFLAGS = -Wall -Werror -pedantic -DMKL_ILP64 -m64 -I${MKLROOT}/include -O3
MKL_FLAGS =  ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -lpthread -lm -ldl
# -Rpass=loop-vectorize (append this to see which for loop is optimized)

#
CFLAGS_LINUX = -Wall -Werror -pedantic  -DMKL_ILP64 -m64 -I${MKLROOT}/include -O3
MKL_FLAGS_LINUX_DY = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

ifndef version
version = macos
endif

ifndef compiler
compiler = clang
endif

.PHONY: all clean

.all: main

%.o : %.c $(DEPS)
	$(CC) -c $< -o $@


main : $(OBJ)
	@mkdir -p $(BDIR)
ifeq ($(version), macos)
	@$(CC)  $(CFLAGS) $(MKL_FLAGS) $^ -o $@
else
	@$(CC) $(CFLAGS_LINUX) $(MKL_FLAGS_LINUX_DY) $^ -o $@
endif
	@mv *.o $(BDIR)

clean :
	rm -rf *.o main $(BDIR)