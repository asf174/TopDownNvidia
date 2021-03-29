"""
Measurements made by the TopDown methodology in back end part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from measure_parts.metric_measure import MetricMeasure, MetricMeasureNvprof, MetricMeasureNsight
from abc import ABC # abstract class
from parameters.back_end_params import BackEndParameters

class BackEnd(MetricMeasure, ABC):
    """Class that defines the Back-End part."""

    pass

class BackEndNvprof(MetricMeasureNvprof, BackEnd):
    """Class that defines the Back-End part with nvprof scan tool."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(BackEndParameters.C_BACK_END_NAME, BackEndParameters.C_BACK_END_DESCRIPTION, 
            BackEndParameters.C_BACK_END_NVPROF_METRICS, BackEndParameters.C_BACK_END_NVPROF_EVENTS, 
            BackEndParameters.C_BACK_END_NVPROF_METRICS, BackEndParameters.C_BACK_END_NVPROF_EVENTS)
        pass

class BackEndNsight(MetricMeasureNsight, BackEnd):
    """Class that defines the BACK-End part with nsight scan tool."""
	
    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(BackEndParameters.C_BACK_END_NAME, BackEndParameters.C_BACK_END_DESCRIPTION,
            BackEndParameters.C_BACK_END_NAME, BackEndParameters.C_BACK_END_DESCRIPTION,
            MetricMeasureParameters.C_BACK_END_NSIGHT_METRICS, MetricMeasureParameters.C_BACK_END_NSIGHT_METRICS)
        pass
