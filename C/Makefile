CC = gcc
CFLAGS = -Wall -Werror -pedantic
DEPS = matrix_op.h tests.h
BDIR = OBJ_FILES
OBJ = main.o matrix_op.o tests.o
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