"""
Measurements made by the TopDown methodology in core bound part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""
import sys
sys.path.insert(1, '../errors/')
sys.path.insert(1, '../parameters/')

from metric_measure_errors import * 
from metric_measure_params import MetricMeasureParameters # TODO IMPORT ONLY ATTRIBUTES
from back_end import BackEnd

class CoreBound(BackEnd):
    """Class that defines the CoreBound (sub-part of BackEnd) part."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        # revisar esta llamada
        super(BackEnd, self).__init__(MetricMeasureParameters.C_CORE_BOUND_NAME, MetricMeasureParameters.C_CORE_BOUND_DESCRIPTION,
            MetricMeasureParameters.C_CORE_BOUND_METRICS, MetricMeasureParameters.C_CORE_BOUND_EVENTS, 
            MetricMeasureParameters.C_CORE_BOUND_METRICS, MetricMeasureParameters.C_CORE_BOUND_EVENTS)
        pass