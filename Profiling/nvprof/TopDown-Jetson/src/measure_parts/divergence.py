"""
Measurements made by the TopDown methodology in divergence part.

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


class Divergence(MetricMeasure):
    """Class that defines the divergence part."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(MetricMeasureParameters.C_DIVERGENCE_NAME, MetricMeasureParameters.C_DIVERGENCE_DESCRIPTION,
            MetricMeasureParameters.C_DIVERGENCE_METRICS, MetricMeasureParameters.C_DIVERGENCE_EVENTS, MetricMeasureParameters.C_DIVERGENCE_METRICS, 
            MetricMeasureParameters.C_DIVERGENCE_EVENTS)
        pass