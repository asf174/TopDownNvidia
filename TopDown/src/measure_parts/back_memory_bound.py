"""
Measurements made by the TopDown methodology in back memory bound part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from parameters.metric_measure_params import MetricMeasureParameters 
from abc import ABC # abstract class
from parameters.back_memory_bound_params import BackCoreBoundParameters
from measure_parts.back_memory_bound import BackCoreBound


class MemoryBound(BackEnd, ABC):
    """Class that defines the Back-End.Memory-Bound part."""
    
    pass

class BackMemoryBoundNsight(MetricMeasureNsight, BackMemoryBound):
    """Class that defines the Back-End.MemoryBound part with nsight scan tool."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super(BackEnd, self).__init__(BackCoreBoundParameters.C_BACK_MEMORY_BOUND_NAME, BackCoreBoundParameters.C_BACK_MEMORY_BOUND_DESCRIPTION,
            BackCoreBoundParameters.C_BACK_MEMORY_BOUND_NSIGHT_METRICS, BackCoreBoundParameters.C_BACK_MEMORY_BOUND_NSIGHT_METRICS, 
            BackCoreBoundParameters.C_BACK_MEMORY_BOUND_NSIGHT_METRICS, BackCoreBoundParameters.C_BACK_MEMORY_BOUND_NSIGHT_METRICS)
        pass

class BackMemoryBoundNvprof(MetricMeasureNvprof, BackMemoryBound):
    """Class that defines the Back-End.MemoryBound part with nvprof scan tool."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super(BackEnd, self).__init__(BackCoreBoundParameters.C_BACK_MEMORY_BOUND_NAME, BackCoreBoundParameters.C_BACK_MEMORY_BOUND_DESCRIPTION,
            BackCoreBoundParameters.C_BACK_MEMORY_BOUND_NVPROF_METRICS, BackCoreBoundParameters.C_BACK_MEMORY_BOUND_NVPROF_METRICS, 
            BackCoreBoundParameters.C_BACK_MEMORY_BOUND_NVPROF_METRICS, BackCoreBoundParameters.C_BACK_MEMORY_BOUND_NVPROF_METRICS)
        pass
   
