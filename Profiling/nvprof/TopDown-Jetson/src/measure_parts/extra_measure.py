"""
Support (extra) meassures of TopDown methodology

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


class ExtraMeasure(MetricMeasure):
    """Class that defines the Front-End part."""

    def __init__(self):
        """Set attributes with DEFAULT values."""

        super().__init__(MetricMeasureParameters.C_EXTRA_MEASURE_NAME, MetricMeasureParameters.C_EXTRA_MEASURE_DESCRIPTION,
            MetricMeasureParameters.C_EXTRA_MEASURE_METRICS, MetricMeasureParameters.C_EXTRA_MEASURE_EVENTS, MetricMeasureParameters.C_EXTRA_MEASURE_METRICS, 
            MetricMeasureParameters.C_EXTRA_MEASURE_EVENTS)
        pass