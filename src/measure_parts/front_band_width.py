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
from measure_parts.front_end import FrontEnd
from measure_parts.metric_measure import MetricMeasureNsight, MetricMeasureNvprof


class FrontBandWidth(FrontEnd, ABC):
    """Class that defines the Front-End.BandWidth part."""
    
    pass

class FrontBandWidthNsight(MetricMeasureNsight, FrontBandWidth):
    """Class that defines the Front-End.BandWidth part with nsight scan tool."""

    def __init__(self, name : str, description : str, metrics : str):
        """
        Set attributtes with argument values.
        
        Params:
            
            name                : str   ;   measure name.
        
            description         : str   ;   description with information.
        
            metrics             : str   ;   string with the metrics
         
        """

        super().__init__(name, description, metrics)
        pass

class FrontBandWidthNvprof(MetricMeasureNvprof, FrontBandWidth):
    """Class that defines the Front-End.BandWidth part with nvprof scan tool."""

    def __init__(self, name : str, description : str, metrics : str, events : str):
        """ 
        Set attributtes with argument values.
        
        Params:
            
            name                : str   ;   measure name.
        
            description         : str   ;   description with information.
        
            metrics             : str   ;   string with the metrics
        
            events              : str   ;   string with events
        """

        super().__init__(name, description, metrics, events)
        pass


