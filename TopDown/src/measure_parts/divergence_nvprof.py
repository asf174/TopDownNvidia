"""
Measurements made by the TopDown methodology in divergence part
with nvprof scan tool.

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

class DivergenceNvprof(MetricMeasureNvprof, Divergence):
    """Class that defines the Divergence part with nvprof scan tool."""

    pass
    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(DivergenceParameters.C_DIVERGENCE_NAME, DivergenceParameters.C_DIVERGENCE_DESCRIPTION,
        DivergenceParameters.C_DIVERGENCE_NVPROF_METRICS, DivergenceParameters.C_DIVERGENCE_NVPROF_EVENTS, 
        DivergenceParameters.C_DIVERGENCE_NVPROF_METRICS, DivergenceParameters.C_DIVERGENCE_NVPROF_EVENTS)
        pass
