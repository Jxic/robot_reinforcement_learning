# for debugging checking memory leaks
CC = /usr/local/Cellar/llvm/8.0.0/bin/clang
CFLAGS = -Wall -Werror -pedantic -O3 -fsanitize=address -g
#LDFLAGS = -L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib -I/usr/local/opt/llvm/include


DEPS = matrix_op.h tests.h utils.h rl.h layers.h data_structures.h macros.h model.h sim_api.h rl_utils.h rl_ddpg.h socket_utils.h optimizer.h model_utils.h
BDIR = OBJ_FILES
OBJ = main.o matrix_op.o tests.o utils.o rl.o layers.o model.o rl_utils.o rl_ddpg.o socket_utils.o optimizer.o model_utils.o sim_api.o
EXE = main

# for production
CC = gcc
CFLAGS = -Wall -Werror -pedantic -DMKL_ILP64 -m64 -I${MKLROOT}/include -O3
MKL_FLAGS =   ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -lpthread -lm -ldl
# -Rpass=loop-vectorize (append this to see which for loop is optimized)

#
CC_LINUX = gcc
CFLAGS_LINUX = -Wall -Werror -pedantic  -DMKL_ILP64 -m64 -I${MKLROOT}/include
MKL_FLAGS_LINUX =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl

.PHONY: all clean

.all:

%.o : %.c $(DEPS)
	$(CC) -c $< -o $@

main_mac : $(OBJ)
	@mkdir -p $(BDIR)
	@$(CC)  $(CFLAGS) $(MKL_FLAGS) $^ -o $@
	@mv *.o $(BDIR)

main_linux: $(OBJ)
	@mkdir -p $(BDIR)
	@$(CC_LINUX)  $(CFLAGS_LINUX) $(MKL_FLAGS_LINUX) $^ -o $@
	@mv *.o $(BDIR)


clean :
	rm -rf *.o main_mac main_linux $(BDIR)