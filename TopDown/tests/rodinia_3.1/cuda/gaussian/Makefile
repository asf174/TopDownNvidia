include ../../common/make.config

CC := /usr/bin/nvcc
CUDA_LIB_DIR=/usr/local/cuda

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian
#KERNEL_DIM
release: $(SRC)
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm gaussian
