"""
Measurements made by the TopDown methodology in front end bandwith part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from abc import ABC # abstract class
from parameters.front_band_width_params import FrontBandWidthParameters
from measure_parts.front_end import FrontEnd
from measure_parts.metric_measure import MetricMeasureNsight, MetricMeasureNvprof


class FrontBandWidth(FrontEnd, ABC):
    """Class that defines the Front-End.BandWidth part."""
    
    pass

class FrontBandWidthNsight(MetricMeasureNsight, FrontBandWidth):
    """Class that defines the Front-End.BandWidth part with nsight scan tool."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NAME, FrontBandWidthParameters.C_FRONT_BAND_WIDTH_DESCRIPTION,
            FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NSIGHT_METRICS, FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NSIGHT_METRICS)
        pass

class FrontBandWidthNvprof(MetricMeasureNvprof, FrontBandWidth):
    """Class that defines the Front-End.BandWidth part with nvprof scan tool."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super().__init__(FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NAME, FrontBandWidthParameters.C_FRONT_BAND_WIDTH_DESCRIPTION,
            FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NVPROF_METRICS, FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NVPROF_EVENTS, 
            FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NVPROF_METRICS, FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NVPROF_EVENTS)
        pass  
