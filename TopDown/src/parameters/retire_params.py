"""
Class with all params of Retire class
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

class RetireParameters:

    C_RETIRE_NAME                    : str       = "RETIRE"
    C_RETIRE_DESCRIPTION             : str       = ("R Description")
    # frond_end_nvprof.py
    C_RETIRE_NVPROF_METRICS          : str       = (LevelExecutionParameters.C_IPC_METRIC_NAME_NVPROF)
    C_RETIRE_NVPROF_EVENTS           : str       = ("")

    # frond_end_nsight.py
    C_RETIRE_NSIGHT_METRICS          : str       = (LevelExecutionParameters.C_IPC_METRIC_NAME_NSIGHT)