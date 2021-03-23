"""
Measurements made by the TopDown methodology in extra measure part
with nsight scan tool.

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

class ExtraMeasureNsight(MetricMeasureNsight, ExtraMeasure):
    """Class that defines the ExtraMeasure part with nsight scan tool."""

    pass
    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(ExtraMeasureParameters.C_EXTRA_MEASURE_NAME, ExtraMeasureParameters.C_EXTRA_MEASURE_DESCRIPTION,
        ExtraMeasureParameters.C_EXTRA_MEASURE_NSIGHT_METRICS, ExtraMeasureParameters.C_EXTRA_MEASURE_NSIGHT_EVENTS, 
        ExtraMeasureParameters.C_EXTRA_MEASURE_NSIGHT_METRICS, ExtraMeasureParameters.C_EXTRA_MEASURE_NSIGHT_EVENTS)
        pass
