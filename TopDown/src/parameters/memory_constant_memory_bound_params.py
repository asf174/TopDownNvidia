"""
Class with all params of BackEnd.MemoryBound.Constant-Memory-Bound class
and their subclasses

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class MemoryConstantMemoryBoundParameters:

    C_MEMORY_CONSTANT_MEMORY_BOUND_NAME                    : str        = "BACK-END.MEMORY-BOUND.CONSTANT-MEMORY-BOUND"
    C_MEMORY_CONSTANT_MEMORY_BOUND_DESCRIPTION             : str        = ("")
    
    # memory_constant_memory_bound_nvprof.py
    C_MEMORY_CONSTANT_MEMORY_BOUND_NVPROF_METRICS          : str        = ("stall_constant_memory_dependency")
    C_MEMORY_CONSTANT_MEMORY_BOUND_NVPROF_EVENTS           : str        = ("")

    # memory_constant_memory_bound_nsight.py
    C_MEMORY_CONSTANT_MEMORY_BOUND_NSIGHT_METRICS          : str        = ("")
