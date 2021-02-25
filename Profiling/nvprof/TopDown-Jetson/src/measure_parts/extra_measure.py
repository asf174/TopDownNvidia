"""
Support (extra) meassures of TopDown methodology

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""
import sys
path : str = "/home/alvaro/Documents/Facultad/"
path_desp : str = "/mnt/HDD/alvaro/"
sys.path.insert(1, path_desp + "TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/errors")
sys.path.insert(1,  path_desp + "TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/parameters")
sys.path.insert(1,  path_desp + "TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/measure_parts")

from metric_measure_errors import * 
from metric_measure_params import MetricMeasureParameters # TODO IMPORT ONLY ATTRIBUTES
from metric_measure import MetricMeasure

class ExtraMeasure(MetricMeasure):
    """Class that defines the Front-End part."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        super().__init__(MetricMeasureParameters.C_EXTRA_MEASURE_NAME, MetricMeasureParameters.C_EXTRA_MEASURE_DESCRIPTION,
            MetricMeasureParameters.C_EXTRA_MEASURE_METRICS, MetricMeasureParameters.C_EXTRA_MEASURE_EVENTS, MetricMeasureParameters.C_EXTRA_MEASURE_METRICS, 
            MetricMeasureParameters.C_EXTRA_MEASURE_EVENTS)
        pass