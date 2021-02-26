"""
Measurements made by the TopDown methodology in memory bound part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

from metric_measure_params import MetricMeasureParameters # TODO IMPORT ONLY ATTRIBUTES
from metric_measure import MetricMeasure
from back_end import BackEnd

class MemoryBound(BackEnd):
    """Class that defines the MemoryBound (sub-part of BackEnd) part."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super(BackEnd, self).__init__(MetricMeasureParameters.C_MEMORY_BOUND_NAME, MetricMeasureParameters.C_MEMORY_BOUND_DESCRIPTION,
            MetricMeasureParameters.C_MEMORY_BOUND_METRICS, MetricMeasureParameters.C_MEMORY_BOUND_EVENTS, 
            MetricMeasureParameters.C_MEMORY_BOUND_METRICS, MetricMeasureParameters.C_MEMORY_BOUND_EVENTS)
        pass