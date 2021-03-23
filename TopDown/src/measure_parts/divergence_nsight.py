"""
Measurements made by the TopDown methodology in divergence part
with nsight scan tool.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from parameters.divergence_end_params import DivergenceParameters 
from measure_parts.divergence import Divergence

class DivergenceNsight(MetricMeasurensight, Divergence):
    """Class that defines the Divergence part with NSIGHT scan tool."""

    pass
    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(DivergenceParameters.C_DIVERGENCE_NAME, DivergenceParameters.C_DIVERGENCE_DESCRIPTION,
        DivergenceParameters.C_DIVERGENCE_NSIGHT_METRICS, DivergenceParameters.C_DIVERGENCE_NSIGHT_EVENTS, 
        DivergenceParameters.C_DIVERGENCE_NSIGHT_METRICS, DivergenceParameters.C_DIVERGENCE_NSIGHT_EVENTS)
        pass
