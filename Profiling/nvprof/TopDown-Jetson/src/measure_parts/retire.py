"""
Measurements made by the TopDown methodology in retire part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from errors.metric_measure_errors import * 
from parameters.metric_measure_params import MetricMeasureParameters # TODO IMPORT ONLY ATTRIBUTES
from measure_parts.metric_measure import MetricMeasure

class Retire(MetricMeasure):
    """Class that defines the Retire part."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(MetricMeasureParameters.C_RETIRE_NAME, MetricMeasureParameters.C_RETIRE_DESCRIPTION,
            MetricMeasureParameters.C_RETIRE_METRICS, MetricMeasureParameters.C_RETIRE_EVENTS, MetricMeasureParameters.C_RETIRE_METRICS, 
            MetricMeasureParameters.C_RETIRE_EVENTS)
        pass
