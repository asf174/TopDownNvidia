"""
Measurements made by the TopDown methodology in extra measure part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from measure_parts.metric_measure import MetricMeasure, MetricMeasureNsight, MetricMeasureNvprof
from abc import ABC # abstract class
from parameters.extra_measure_params import ExtraMeasureParameters 

class ExtraMeasure(MetricMeasure, ABC):
    """Class that defines the ExtraMeasure part."""

    pass

class ExtraMeasureNsight(MetricMeasureNsight, ExtraMeasure):
    """Class that defines the ExtraMeasure part with nsight scan tool."""

    pass
    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(ExtraMeasureParameters.C_EXTRA_MEASURE_NAME, ExtraMeasureParameters.C_EXTRA_MEASURE_DESCRIPTION,
        ExtraMeasureParameters.C_EXTRA_MEASURE_NSIGHT_METRICS, ExtraMeasureParameters.C_EXTRA_MEASURE_NSIGHT_METRICS)
        pass

class ExtraMeasureNvprof(MetricMeasureNvprof, ExtraMeasure):
    """Class that defines the ExtraMeasure part with nvprof scan tool."""

    pass
    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(ExtraMeasureParameters.C_EXTRA_MEASURE_NAME, ExtraMeasureParameters.C_EXTRA_MEASURE_DESCRIPTION,
        ExtraMeasureParameters.C_EXTRA_MEASURE_NVPROF_METRICS, ExtraMeasureParameters.C_EXTRA_MEASURE_NVPROF_EVENTS, 
        ExtraMeasureParameters.C_EXTRA_MEASURE_NVPROF_METRICS, ExtraMeasureParameters.C_EXTRA_MEASURE_NVPROF_EVENTS)
        pass
   
