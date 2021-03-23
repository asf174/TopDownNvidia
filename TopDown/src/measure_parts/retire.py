"""
Measurements made by the TopDown methodology in retire part
with nsight scan tool.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from parameters.retire_end_params import RetireParameters 
from measure_parts.retire import Retire

class Retire(MetricMeasure, ABC)):
    """Class that defines the Retire part."""

    pass
