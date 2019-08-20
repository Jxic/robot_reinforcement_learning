
CXX = g++
CL_COMPILER = aoc
ARCH_FLAG = -march=emulator
BOARD_FLAG = -board=a10gx


INC = -I/opt/intelFPGA_pro/17.1/hld/host/include
LIB = -L/opt/intelFPGA_pro/17.1/hld/board/a10_ref/linux64/lib -L/opt/intelFPGA_pro/17.1/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl -laltera_a10_ref_mmd -lelf -lrt -lpthread

CXXFLAGS = -Wall  $(INC)

DEPS = setup.hpp tests.hpp utils.hpp c_interface.hpp
OBJ = setup.o main.o tests.o utils.o c_interface.o
EXE = main

.PHONY: all clean

all: main rl.aocx

%o : %.c $(DEPS)
	$(CC) -c -o $@ $< $(CXXFLAGS) 

main : $(OBJ)
	$(CC) $(LIB) $(CFLAGS) $^ -o $@

rl.aocx: rl.cl
	$(CL_COMPILER) $(ARCH_FLAG) $(BOARD_FLAG) $<
	

clean :
	rm -rf *.o main *.aocx *.aoco rl

# CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1
