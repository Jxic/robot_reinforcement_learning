CC = gcc
CFLAGS = -Wall -Werror -pedantic -O3
DEPS = matrix_op.h tests.h utils.h rl.h layers.h data_structures.h macros.h model.h sim_api.h rl_utils.h rl_ddpg.h
BDIR = OBJ_FILES
OBJ = main.o matrix_op.o tests.o utils.o rl.o layers.o model.o rl_utils.o rl_ddpg.o
EXE = main

.PHONY: all clean

.all: $(EXE)

%.o : %.c $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

main : $(OBJ)
	@mkdir -p $(BDIR)
	@$(CC) $(CFLAGS) $^ -o $@
	@mv *.o $(BDIR)

clean :
	rm -rf *.o main $(BDIR)