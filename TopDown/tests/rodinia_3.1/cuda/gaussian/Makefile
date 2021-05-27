include ../../common/make.config

CC := /usr/bin/nvcc
CUDA_LIB_DIR=/usr/bin/cuda

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm gaussian
