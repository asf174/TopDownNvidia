#!/bin/bash
sudo $(which nvprof) --events ipc --unified-memory-profiling off ../../../CUDA/bin/add_two_matrix 2> $1
