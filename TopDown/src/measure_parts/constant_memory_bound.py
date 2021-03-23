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
from parameters.metric_measure_params import MetricMeasureParameters
from measure_parts.memory_bound import MemoryBound
from measure_parts.back_end import BackEnd

class ConstantMemoryBound(MemoryBound):
    """Class that defines the ConstantMemoryBound (sub-part of MemoryBound) part."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super(BackEnd, self).__init__(MetricMeasureParameters.C_CONSTANT_MEMORY_BOUND_NAME, 
            MetricMeasureParameters.C_CONSTANT_MEMORY_BOUND_DESCRIPTION, MetricMeasureParameters.C_CONSTANT_MEMORY_BOUND_METRICS, 
            MetricMeasureParameters.C_CONSTANT_MEMORY_BOUND_EVENTS, MetricMeasureParameters.C_CONSTANT_MEMORY_BOUND_METRICS, 
            MetricMeasureParameters.C_CONSTANT_MEMORY_BOUND_EVENTS)
        pass