"""
Measurements made by the TopDown methodology in back end part
with nsight scan tool.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from parameters.front_end_params import BackEndParameters 
from measure_parts.front_end import BackEnd

class BackEndNsight(MetricMeasureNvprof, BackEnd):
    """Class that defines the Front-End part with nsight scan tool."""

    pass
    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(BackEndParameters.C_FRONT_END_NAME, BackEndParameters.C_FRONT_END_DESCRIPTION,
        BackEndParameters.C_FRONT_END_NAME, BackEndParameters.C_FRONT_END_DESCRIPTION,
        MetricMeasureParameters.C_FRONT_END_NSIGHT_METRICS, MetricMeasureParameters.C_FRONT_END_NSIGHT_METRICS)
        pass