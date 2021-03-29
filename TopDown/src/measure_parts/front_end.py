"""
Measurements made by the TopDown methodology in front end part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from parameters.front_end_params import FrontEndParameters 
from measure_parts.metric_measure import MetricMeasure, MetricMeasureNsight, MetricMeasureNvprof
from abc import ABC # abstract class
from parameters.front_end_params import FrontEndParameters 

class FrontEnd(MetricMeasure, ABC):
    """Class that defines the Front-End part."""
    
    pass
 
class FrontEndNsight(MetricMeasureNsight, FrontEnd):
    """Class that defines the Front-End part with nsight scan tool."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
            
        super(MetricMeasureNsight, self).__init__(FrontEndParameters.C_FRONT_END_NAME, FrontEndParameters.C_FRONT_END_DESCRIPTION,
            MetricMeasureParameters.C_FRONT_END_NSIGHT_METRICS, MetricMeasureParameters.C_FRONT_END_NSIGHT_METRICS)
        pass
                   
class FrontEndNvprof(MetricMeasureNvprof, FrontEnd):
    """Class that defines the Front-End part with nvprof scan tool."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super(FrontEndNvprof, self).__init__(FrontEndParameters.C_FRONT_END_NAME, FrontEndParameters.C_FRONT_END_DESCRIPTION,
            FrontEndParameters.C_FRONT_END_NVPROF_METRICS, FrontEndParameters.C_FRONT_END_NVPROF_EVENTS, 
            FrontEndParameters.C_FRONT_END_NVPROF_METRICS, FrontEndParameters.C_FRONT_END_NVPROF_EVENTS)
        pass

