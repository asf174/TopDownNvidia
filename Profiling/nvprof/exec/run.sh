#!/bin/bash
sudo $(which nvprof) --metrics ipc --unified-memory-profiling off ../../../CUDA/bin/add_two_matrix -D N=500 2> $1
