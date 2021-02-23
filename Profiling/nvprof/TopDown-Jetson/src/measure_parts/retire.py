"""
Measurements made by the TopDown methodology in retire part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import sys
sys.path.insert(1, '/mnt/HDD/alvaro/TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/errors')
sys.path.insert(1, '/mnt/HDD/alvaro/TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/parameters')
sys.path.insert(1, '/mnt/HDD/alvaro/TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/measure')

from metric_measure_errors import * 
from metric_measure_params import MetricMeasureParameters # TODO IMPORT ONLY ATTRIBUTES
from metric_measure import MetricMeasure

class Retire(MetricMeasure):
    """Class that defines the Retire part."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        super().__init__(MetricMeasureParameters.C_RETIRE_NAME, MetricMeasureParameters.C_RETIRE_DESCRIPTION,
            MetricMeasureParameters.C_RETIRE_METRICS, MetricMeasureParameters.C_RETIRE_EVENTS, MetricMeasureParameters.C_RETIRE_METRICS, 
            MetricMeasureParameters.C_RETIRE_EVENTS)
        pass
