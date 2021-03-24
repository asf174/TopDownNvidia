"""
Measurements made by the TopDown methodology in back core bound part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from parameters.metric_measure_params import MetricMeasureParameters 
from abc import ABC # abstract class

class CoreBound(BackEnd, ABC):
    """Class that defines the Back-End.Core-Bound part."""
    
    pass
   
# TODO revisar este constructor.