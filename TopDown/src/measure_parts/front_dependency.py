"""
Measurements made by the TopDown methodology in FrontEnd's dependency limit part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from parameters.metric_measure_params import MetricMeasureParameters 
from measure_parts.front_end import FrontEnd
from abc import ABC # abstract class

class FrontDependency(FrontEnd, ABC):
    """Class that defines the Front-End.Dependency part."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super(FrontEnd, self).__init__(MetricMeasureParameters.C_FRONT_DEPENDENCY_NAME, MetricMeasureParameters.C_FRONT_DEPENDENCY_DESCRIPTION,
            MetricMeasureParameters.C_FRONT_DEPENDENCY_METRICS, MetricMeasureParameters.C_FRONT_DEPENDENCY_EVENTS, 
            MetricMeasureParameters.C_FRONT_DEPENDENCY_METRICS, MetricMeasureParameters.C_FRONT_DEPENDENCY_EVENTS)
        pass