"""
Measurements made by the TopDown methodology in extra measure part
with nvprof scan tool.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from parameters.extra_measure_end_params import ExtraMeasureParameters 
from measure_parts.extra_measure import ExtraMeasure

class ExtraMeasureNvprof(MetricMeasureNvprof, ExtraMeasure):
    """Class that defines the ExtraMeasure part with nvprof scan tool."""

    pass
    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(ExtraMeasureParameters.C_EXTRA_MEASURE_NAME, ExtraMeasureParameters.C_EXTRA_MEASURE_DESCRIPTION,
        ExtraMeasureParameters.C_EXTRA_MEASURE_NVPROF_METRICS, ExtraMeasureParameters.C_EXTRA_MEASURE_NVPROF_EVENTS, 
        ExtraMeasureParameters.C_EXTRA_MEASURE_NVPROF_METRICS, ExtraMeasureParameters.C_EXTRA_MEASURE_NVPROF_EVENTS)
        pass
