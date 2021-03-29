"""
Class with all params of BackMEMORY_BOUND-MemoryBound class
and their subclasses

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class BackMemoryBoundParameters:

    C_BACK_MEMORY_BOUND_NAME                    : str        = "BACK-END.MEMORY-BOUND"
    C_BACK_MEMORY_BOUND_DESCRIPTION             : str        = ("It analyzes the parts of the GPU architecture where we have a loss of performance (IPC) due to\n"
                                                                + "memory bounds. This part takes into account aspects such as data dependencies, failures or access\n"
                                                                + "limits in caches")
    # back_MEMORY_BOUND_nvprof.py
    C_BACK_MEMORY_BOUND_NVPROF_METRICS          : str        =  ("stall_memory_dependency,stall_constant_memory_dependency,stall_memory_throttle")
    C_BACK_MEMORY_BOUND_NVPROF_EVENTS           : str        = ("")

    # back_MEMORY_BOUND_nsight.py
    C_BACK_MEMORY_BOUND_NSIGHT_METRICS          : str        = ("")