"""
Measurements made by the TopDown methodology in divergence part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""
import sys
sys.path.insert(1, '../errors/')
sys.path.insert(1, '../parameters/')

from metric_measure_errors import * 
from metric_measure_params import MetricMeasureParameters # TODO IMPORT ONLY ATTRIBUTES
from metric_measure import MetricMeasure

class Divergence(MetricMeasure):
    """Class that defines the divergence part."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        super().__init__(MetricMeasureParameters.C_DIVERGENCE_NAME, MetricMeasureParameters.C_DIVERGENCE_DESCRIPTION,
            MetricMeasureParameters.C_DIVERGENCE_METRICS, MetricMeasureParameters.C_DIVERGENCE_EVENTS, MetricMeasureParameters.C_DIVERGENCE_METRICS, 
            MetricMeasureParameters.C_DIVERGENCE_EVENTS)
        pass