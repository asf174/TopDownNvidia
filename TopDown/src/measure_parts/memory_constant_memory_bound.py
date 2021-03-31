"""
Measurements made by the TopDown methodology in constant memory part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from measure_parts.back_memory_bound import BackMemoryBound
from measure_parts.back_end import BackEnd
from parameters.memory_constant_memory_bound_params import MemoryConstantMemoryBoundParameters 
from measure_parts.back_memory_bound import BackMemoryBound
from measure_parts.metric_measure import MetricMeasureNsight, MetricMeasureNvprof
 
class MemoryConstantMemoryBound(BackMemoryBound):
    """Class that defines the ConstantMemoryBound (sub-part of MemoryBound) part."""

    pass
 
class MemoryConstantMemoryBoundNsight(MetricMeasureNsight, MemoryConstantMemoryBound):
    """Class that defines the Back-End.MemoryCound.Constant-Memory-Bound part with nsight scan tool."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NAME, MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_DESCRIPTION,
            MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NSIGHT_METRICS, MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NSIGHT_METRICS)
        pass   

class MemoryConstantMemoryBoundNvprof(MetricMeasureNvprof, MemoryConstantMemoryBound):
    """Class that defines the Back-End.CoreBound part with nvprof scan tool."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NAME, MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_DESCRIPTION,
            MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NVPROF_METRICS, MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NVPROF_METRICS, 
            MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NVPROF_METRICS, MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NVPROF_METRICS)
        pass       
