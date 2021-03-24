"""
Class with all params of ExtraMeasure class
and their subclasses

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from parameters.level_execution_params import LevelExecutionParameters

class ExtraMeasureParameters:

    C_EXTRA_MEASURE_NAME                    : str      = "EXTRA-MEASURE"
    C_EXTRA_MEASURE_DESCRIPTION             : str      = ("")
    
    # extra_measure_nvprof.py
    C_EXTRA_MEASURE_NVPROF_METRICS          : str      = ("")
    C_EXTRA_MEASURE_NVPROF_EVENTS           : str      = ("active_cycles," + LevelExecutionParameters.C_CYCLES_ELAPSED_NAME)

    # extra_measure_nsight.py
    C_EXTRA_MEASURE_NSIGHT_METRICS          : str      = ("")