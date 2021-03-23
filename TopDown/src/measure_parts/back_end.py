"""
Measurements made by the TopDown methodology in back end part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from measure_parts.metric_measure import MetricMeasure
from abc import ABC # abstract class

class BackEnd(MetricMeasure, ABC):
    """Class that defines the Back-End part."""

    pass