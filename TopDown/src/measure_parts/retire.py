"""
Measurements made by the TopDown methodology in retire part
with nsight scan tool.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from parameters.retire_params import RetireParameters 
from abc import ABC # abstract class
from measure_parts.metric_measure import MetricMeasure, MetricMeasureNsight, MetricMeasureNvprof

class Retire(MetricMeasure, ABC):
    """Class that defines the Retire part."""

    pass

class RetireNsight(MetricMeasureNsight, Retire):
    """Class that defines the Retire part with nsight scan tool."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(RetireParameters.C_RETIRE_NAME, RetireParameters.C_RETIRE_DESCRIPTION,
        RetireParameters.C_RETIRE_NSIGHT_METRICS, RetireParameters.C_RETIRE_NSIGHT_EVENTS, 
        RetireParameters.C_RETIRE_NSIGHT_METRICS, RetireParameters.C_RETIRE_NSIGHT_EVENTS)
        pass

class RetireNvprof(MetricMeasureNvprof, Retire):
    """Class that defines the Retire part with NVPROF scan tool."""
  
    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(RetireParameters.C_RETIRE_NAME, RetireParameters.C_RETIRE_DESCRIPTION,
        RetireParameters.C_RETIRE_NVPROF_METRICS, RetireParameters.C_RETIRE_NVPROF_EVENTS, 
        RetireParameters.C_RETIRE_NVPROF_METRICS, RetireParameters.C_RETIRE_NVPROF_EVENTS)
        pass

