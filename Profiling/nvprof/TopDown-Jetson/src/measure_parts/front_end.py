"""
Measurements made by the TopDown methodology in front end part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""
import sys
path : str = "/home/alvaro/Documents/Facultad/"
path_desp : str = "/mnt/HDD/alvaro/"
sys.path.insert(1, path_desp + "TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/errors")
sys.path.insert(1,  path_desp + "TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/parameters")

from metric_measure_errors import * 
from metric_measure_params import MetricMeasureParameters # TODO IMPORT ONLY ATTRIBUTES
from metric_measure import MetricMeasure

class FrontEnd(MetricMeasure):
    """Class that defines the Front-End part."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        super().__init__(MetricMeasureParameters.C_FRONT_END_NAME, MetricMeasureParameters.C_FRONT_END_DESCRIPTION,
            MetricMeasureParameters.C_FRONT_END_METRICS, MetricMeasureParameters.C_FRONT_END_EVENTS, MetricMeasureParameters.C_FRONT_END_METRICS, 
            MetricMeasureParameters.C_FRONT_END_EVENTS)
        pass