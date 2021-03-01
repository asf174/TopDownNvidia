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

from parameters.metric_measure_params import MetricMeasureParameters # TODO IMPORT ONLY ATTRIBUTES
from measure_parts.metric_measure import MetricMeasure


class BackEnd(MetricMeasure):
    """Class that defines the Back-End part."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(MetricMeasureParameters.C_BACK_END_NAME, MetricMeasureParameters.C_BACK_END_DESCRIPTION,
            MetricMeasureParameters.C_BACK_END_METRICS, MetricMeasureParameters.C_BACK_END_EVENTS, MetricMeasureParameters.C_BACK_END_METRICS, 
            MetricMeasureParameters.C_BACK_END_EVENTS)
        pass