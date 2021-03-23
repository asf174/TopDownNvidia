#!/bin/bash
#sudo $(which nvprof) --metrics ipc --unified-memory-profiling off ../../../CUDA/bin/add_two_matrix -D N=500 2> $1
#sudo $(which nvprof) --metrics inst_per_warp,inst_executed,ipc --events warps_launched,elapsed_cycles_sm --unified-memory-profiling off --profile-from-start off ../../../CUDA/bin/add_two_matrix2
#sudo $(which nvprof) --metrics inst_per_warp,inst_executed,inst_issued,ipc --events elapsed_cycles_sm,active_cycles --unified-memory-profiling off --profile-from-start off ../../../CUDA/bin/hello_world

#sudo $(which nvprof) --metrics stall_inst_fetch --unified-memory-profiling off --profile-from-start off ../../../CUDA/bin/add_two_matrix2
#sudo $(which nvprof) --metrics stall_exec_dependency --unified-memory-profiling off --profile-from-start off ../../../CUDA/bin/add_two_matrix2
#sudo $(which nvprof) --metrics stall_sync --unified-memory-profiling off --profile-from-start off ../../../CUDA/bin/add_two_matrix2
#sudo $(which nvprof) --metrics stall_other --unified-memory-profiling off --profile-from-start off ../../../CUDA/bin/add_two_matrix2
#sudo $(which nvprof) --metrics stall_not_selected --unified-memory-profiling off --profile-from-start off ../../../CUDA/bin/add_two_matrix2

sudo $(which nvprof) --metrics stall_inst_fetch,stall_exec_dependency,stall_sync,stall_other,stall_not_selected,stall_not_selected --unified-memory-profiling off --profile-from-start off ../../../../CUDA/bin/add_two_matrix2
