# Simple Makefile to compile all files in src/ directory
CC=/usr/local/cuda-9.0/bin/nvcc
flags= -Wall -Werror
dirCU=src/
dirB=bin/


all: add_two_array2.o add_two_matrix.o add_two.o hello_world.o
	$(CC) -o $(dirB)add_two_array2 $(dirB)add_two_array2.o
	#$(CC) -o $(dirB)add_two_array $(dirB)add_two_array.o
	$(CC) -o $(dirB)add_two_matrix $(dirB)add_two_matrix.o
	$(CC) -o $(dirB)add_two $(dirB)add_two.o
	$(CC) -o $(dirB)hello_world $(dirB)hello_world.o

add_two_array2.o: src/add_two_array2.cu
	$(CC) -c $(dirCU)add_two_array2.cu -o $(dirB)add_two_array2.o

#add_two_array.o: $(dirCU)add_two_array.cu
	#$(CC) -c $(dirCU)add_two_array.cu -o $(dirB)add_two_array.o

add_two_matrix.o: $(dirCU)add_two_matrix.cu
	$(CC) -c $(dirCU)add_two_matrix.cu -o $(dirB)add_two_matrix.o

add_two.o: $(dirCU)add_two.cu
	$(CC) -c $(dirCU)add_two.cu -o $(dirB)add_two.o

hello_world.o: $(dirCU)hello_world.cu
	$(CC) -c $(dirCU)hello_world.cu -o $(dirB)hello_world.o


.PHONY: clean
