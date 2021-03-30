"""
Class with all params of Divergence class
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

class DivergenceParameters:

    C_DIVERGENCE_NAME                    : str      = "DIVERGENCE"
    C_DIVERGENCE_DESCRIPTION             : str      = ("It analyzes the parts of the GPU architecture where divergence causes a loss of performance.\n" + 
                                                        "This problem is caused when warps are not used correctly. This is caused for example when, for example, there are\n" + 
                                                        "threads (of the same warp) that have to execute an instruction and others do not. In this case there are GPU cores\n" + 
                                                        "that are not being used. Another worst case occurs when some threads of a warp have to execute one instruction and\n" + 
                                                        "others another (if-else). In this case, twice as many cycles are necessary to execute, and in all cases part of the\n" +  
                                                        "cores will not be used.") # TODO preguntar separaciones, si anhadir el espacio
    # divergence_nvprof.py
    C_DIVERGENCE_NVPROF_METRICS          : str      = ("branch_efficiency," + LevelExecutionParameters.C_WARP_EXECUTION_EFFICIENCY_METRIC_NAME_NVPROF + "," + 
                                                      LevelExecutionParameters.C_ISSUE_IPC_METRIC_NAME_NVPROF)
    C_DIVERGENCE_NVPROF_EVENTS           : str      = ("branch,divergent_branch")

    # divergence_nsight.py
    C_DIVERGENCE_NSIGHT_METRICS          : str      = ("smsp__sass_average_branch_targets_threads_uniform.pct," + LevelExecutionParameters.C_WARP_EXECUTION_EFFICIENCY_METRIC_NAME_NSIGHT + 
                                                      "," + LevelExecutionParameters.C_ISSUE_IPC_METRIC_NAME_NSIGHT)
