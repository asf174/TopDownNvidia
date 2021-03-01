"""
Measurements made by the TopDown methodology in FrontEnd's bandwith limits part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from parameters.metric_measure_params import MetricMeasureParameters # TODO IMPORT ONLY ATTRIBUTES
from measure_parts.front_end import FrontEnd

class FrontBandWidth(FrontEnd):
    """Class that defines the Front-End.Band_width part."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super(FrontEnd, self).__init__(MetricMeasureParameters.C_FRONT_BAND_WIDTH_NAME, MetricMeasureParameters.C_FRONT_BAND_WIDTH_DESCRIPTION,
            MetricMeasureParameters.C_FRONT_BAND_WIDTH_METRICS, MetricMeasureParameters.C_FRONT_BAND_WIDTH_EVENTS, MetricMeasureParameters.C_FRONT_BAND_WIDTH_METRICS, 
            MetricMeasureParameters.C_FRONT_BAND_WIDTH_EVENTS)
        pass