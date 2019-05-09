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
CFLAGS = -Wall -Werror -pedantic -O3

.PHONY: all clean

.all: $(EXE)

%.o : %.c $(DEPS)
	$(CC)  $(CFLAGS) -c $< -o $@

main : $(OBJ)
	@mkdir -p $(BDIR)
	@$(CC)  $(CFLAGS) $^ -o $@
	@mv *.o $(BDIR)

clean :
	rm -rf *.o main $(BDIR)