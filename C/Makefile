# for debugging checking memory leaks
CC = /usr/local/Cellar/llvm/8.0.0/bin/clang
CFLAGS = -fsanitize=address -g

DEPS = matrix_op.h tests.h utils.h rl.h layers.h data_structures.h macros.h model.h sim_api.h rl_utils.h rl_ddpg.h socket_utils.h optimizer.h model_utils.h rl_ddpg_her.h normalizer.h test_agent.h rl_ddpg_her_sim.h rl_ddpg_her_demo.h rl_ddpg_her_demo_sim.h multi_agents/rl_ddpg_her_mpi.h multi_agents/mpi_utils.h
BDIR = OBJ_FILES
OBJ = main.o matrix_op.o tests.o utils.o rl.o layers.o model.o rl_utils.o rl_ddpg.o socket_utils.o optimizer.o model_utils.o sim_api.o rl_ddpg_her.o normalizer.o test_agent.c rl_ddpg_her_sim.o rl_ddpg_her_demo.o rl_ddpg_her_demo_sim.o multi_agents/rl_ddpg_her_mpi.o multi_agents/mpi_utils.o
EXE = main

# for production
CC = mpicc
CFLAGS = -Wall -Werror -pedantic -DMKL_ILP64 -m64 -I${MKLROOT}/include  -O3
MKL_FLAGS =  ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -lpthread -lm -ldl
#-Rpass=loop-vectorize -Rpass-analysis=loop-vectorize (append this to see which for loop is optimized)

#
CFLAGS_LINUX = -Wall -Werror -pedantic  -DMKL_ILP64 -m64 -I${MKLROOT}/include -O3
MKL_FLAGS_LINUX_DY =   -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

GPU_DY = -L/usr/local/cuda/lib64 -lcublas -lcudart

ifndef version
version = macos
endif

ifndef compiler
compiler = clang
endif

.PHONY: all clean

.all: main

%.o : %.c $(DEPS)
	$(CC) -c $< -o $@ -pg


main : $(OBJ)
	@mkdir -p $(BDIR)
ifeq ($(version), macos)
	@$(CC)  $(CFLAGS) $(MKL_FLAGS) $^ -o $@
else
	@$(CC) $(CFLAGS_LINUX)  $^ -o $@ $(MKL_FLAGS_LINUX_DY) -pg
endif
	@mv *.o $(BDIR)
	@mv multi_agents/*.o $(BDIR)

clean :
	rm -rf *.o multi_agents/*.o main $(BDIR)
