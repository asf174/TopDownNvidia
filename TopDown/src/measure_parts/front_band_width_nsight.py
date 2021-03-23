"""
Measurements made by the TopDown methodology in FrontEnd's bandwith limits part
with nsight scan tool.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from parameters.front_band_width import FrontBandWidth 
from measure_parts.front_band_width import FrontBandWidth


class FrontBandWidthNsight(MetricMeasureNsight, FrontBandWidth):
    """Class that defines the Front-End.BandWidth part with nsight scan tool."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super(FrontEnd, self).__init__(FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NAME, FrontBandWidthParameters.C_FRONT_BAND_WIDTH_DESCRIPTION,
            FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NSIGHT_METRICS, FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NSIGHT_METRICS, 
            FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NSIGHT_METRICS, FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NSIGHT_METRICS)
        pass