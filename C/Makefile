# for debugging checking memory leaks
CC = /usr/local/Cellar/llvm/8.0.0/bin/clang
# CFLAGS = -fsanitize=address -g

DEPS = matrix_op.h tests.h utils.h rl.h layers.h data_structures.h macros.h model.h sim_api.h rl_utils.h rl_ddpg.h socket_utils.h optimizer.h model_utils.h rl_ddpg_her.h normalizer.h test_agent.h rl_ddpg_her_sim.h rl_ddpg_her_demo.h rl_ddpg_her_demo_sim.h multi_agents/rl_ddpg_her_mpi.h multi_agents/mpi_utils.h rl_ddpg_pixel.c rl_deep_q.h
BDIR = OBJ_FILES
OBJ = main.o matrix_op.o tests.o utils.o rl.o layers.o model.o rl_utils.o rl_ddpg.o socket_utils.o optimizer.o model_utils.o sim_api.o rl_ddpg_her.o normalizer.o test_agent.o rl_ddpg_her_sim.o rl_ddpg_her_demo.o rl_ddpg_her_demo_sim.o multi_agents/rl_ddpg_her_mpi.o multi_agents/mpi_utils.o rl_ddpg_pixel.o rl_deep_q.o
EXE = main

# for production
CC = gcc
CFLAGS = -Wall -DMKL_ILP64 -m64 -I${MKLROOT}/include -O3 -g
MKL_FLAGS = ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_intel_thread.a ${MKLROOT}/lib/libmkl_core.a -liomp5 -lpthread -lm -ldl

CXX = g++
CL_COMPILER = aoc
ARCH_FLAG = -march=emulator
BOARD_FLAG = -board=a10gx

# -I/opt/intelFPGA_pro/17.1/hld/host/include
INC = $(shell aocl compile-config )
# -L/opt/intelFPGA_pro/17.1/hld/board/a10_ref/linux64/lib -L/opt/intelFPGA_pro/17.1/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl -laltera_a10_ref_mmd -lelf -lrt -lpthread
LIB = $(shell aocl link-config )

CXXFLAGS = -Wall  $(INC) -g -std=c++11
CXXDEPS = device/setup.hpp device/tests.hpp device/utils.hpp device/c_interface.hpp
CXXOBJS = device/setup.cxx.o device/main.cxx.o device/tests.cxx.o device/utils.cxx.o device/c_interface.cxx.o

#-Rpass=loop-vectorize -Rpass-analysis=loop-vectorize (append this to see which for loop is optimized)
CFLAGS_LINUX = -Wall -Werror -pedantic  -DMKL_ILP64 -m64 -I${MKLROOT}/include -O3
MKL_FLAGS_LINUX_DY = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

LINUX_DY = $(MKL_FLAGS_LINUX_DY) $(LIB)

GPU_DY = -L/usr/local/cuda/lib64 -lcublas -lcudart

EXE = main

ifeq ($(shell uname), Linux)
# 1 Linux, 0 macos
CFLAGS += -D OPENCL
EXE += rl.aocx
OBJ += $(CXXOBJS)
version = 1
else
version = 0
endif

.PHONY: all clean

all: $(EXE)
	@echo $(EXE)

%.o : %.c $(DEPS)
	$(CC) -c $< -o $@ $(CFLAGS)

%.cxx.o : %.cpp $(CXXDEPS)
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INC)

main : $(OBJ)
	@mkdir -p $(BDIR)
ifeq ($(version), 0)
	@$(CC)  $^ -o $@ $(MKL_FLAGS) -fsanitize=address -g
else
	@$(CXX) $^ -o $@ $(LINUX_DY)
endif
	@mv *.o $(BDIR)
	# @mv device/*.cxx.o $(BDIR)
	@mv multi_agents/*.o $(BDIR)

rl.aocx: device/rl.cl
	$(CL_COMPILER) $(ARCH_FLAG) $(BOARD_FLAG) $<

clean :
	rm -rf *.o multi_agents/*.o $(BDIR) *.aocx *.aoco rl $(EXE) device/*.cxx.o
