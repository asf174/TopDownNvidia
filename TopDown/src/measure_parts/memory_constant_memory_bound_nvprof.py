"""
Measurements made by the TopDown methodology in BackEnd's constant memory bound limits part
with nvprof scan tool.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from parameters.back_core_bound_params import BackCoreBound 
from measure_parts.back_core_bound import BackCoreBound


class ConstantMemoryBoundNvprof(MetricMeasureNvprof, ConstantMemoryBound):
    """Class that defines the Back-End.CoreBound part with nvprof scan tool."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super(BackEnd, self).__init__(MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NAME, MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_DESCRIPTION,
            MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NVPROF_METRICS, MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NVPROF_METRICS, 
            MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NVPROF_METRICS, MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NVPROF_METRICS)
        pass     
