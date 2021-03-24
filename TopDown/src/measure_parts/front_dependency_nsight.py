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
from parameters.front_dependency_params import FrontDependency 
from measure_parts.front_dependency import FrontDependency


class FrontDependencyNsight(MetricMeasureNsight, FrontDependency):
    """Class that defines the Front-End.Dependency part with nsight scan tool."""

    def __init__(self):
        """Set attributes with DEFAULT values."""
        
        super(FrontEnd, self).__init__(FrontDependencyParameters.C_FRONT_DEPENDENCY_NAME, FrontDependencyParameters.C_FRONT_DEPENDENCY_DESCRIPTION,
            FrontDependencyParameters.C_FRONT_DEPENDENCY_NSIGHT_METRICS, FrontDependencyParameters.C_FRONT_DEPENDENCY_NSIGHT_METRICS, 
            FrontDependencyParameters.C_FRONT_DEPENDENCY_NSIGHT_METRICS, FrontDependencyParameters.C_FRONT_DEPENDENCY_NSIGHT_METRICS)
        pass