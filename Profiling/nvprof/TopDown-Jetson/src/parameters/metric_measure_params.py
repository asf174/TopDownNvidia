"""
Class with all params of MetricMetricMeasure class
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

class MetricMeasureParameters:


    # front_end.py
    C_FRONT_END_METRICS                 : str       = ("stall_inst_fetch,stall_exec_dependency,stall_sync,stall_other," +
                                                        "stall_not_selected,stall_not_selected")
    C_FRONT_END_EVENTS                  : str       = ("")
    C_FRONT_END_NAME                    : str       = "FRONT-END"
    C_FRONT_END_DESCRIPTION             : str       = ("It analyzes the parts of the GPU architecture where the FrontEnd produces bottlenecks,\n" 
                                                    + "which leads to IPC losses. In this part, aspects related to the fetch of instructions\n"
                                                    + "are analyzed, such as errors in the instruction cache or IPC losses due to thread synchronization.\n")

    # back_end.py
    C_BACK_END_METRICS                  : str       = ("stall_memory_dependency,stall_constant_memory_dependency,stall_pipe_busy," +
                                                        "stall_memory_throttle")
    C_BACK_END_EVENTS                   : str       = ("")
    C_BACK_END_NAME                     : str       = "BACK-END"
    C_BACK_END_DESCRIPTION              : str       = ("It analyzes the parts of the GPU architecture where the BackEnd produces bottleneck,\n"
                                                    + "which leads to IPC losses. In this part, We analyze aspects related to the 'execution' part of\n"
                                                    + "the instructions, in which aspects such as limitations by functional units, memory limits, etc.\n")

    # divergence.py
    C_DIVERGENCE_METRICS                : str       = ("branch_efficiency,warp_execution_efficiency,issued_ipc")
    C_DIVERGENCE_EVENTS                 : str       = ("branch,divergent_branch")
    C_DIVERGENCE_NAME                   : str       = "DIVERGENCE"
    C_DIVERGENCE_DESCRIPTION            : str       = ("It analyzes the parts of the GPU architecture where divergence causes a loss of performance.\n" + 
                                                        "This problem is caused when warps are not used correctly. This is caused for example when, for example, there are\n" + 
                                                        "threads (of the same warp) that have to execute an instruction and others do not. In this case there are GPU cores\n" + 
                                                        "that are not being used. Another worst case occurs when some threads of a warp have to execute one instruction and\n" + 
                                                        "others another (if-else). In this case, twice as many cycles are necessary to execute, and in all cases part of the\n" +  
                                                        "cores will not be used.") # TODO preguntar separaciones, si anhadir el espacio

    # extra_measure.py
    C_EXTRA_MEASURE_METRICS             : str       = ("")
    C_EXTRA_MEASURE_EVENTS              : str       = ("inst_issued,ins_executed,active_cycles," + LevelExecutionParameters.C_CYCLES_ELAPSED_NAME)
    C_EXTRA_MEASURE_NAME                : str       = "EXTRA_MEASURE"
    C_EXTRA_MEASURE_DESCRIPTION         : str       = ("") # TODO preguntar separaciones, si anhadir el espacio

    # retire.py
    C_RETIRE_METRICS                    : str       = ("ipc")
    C_RETIRE_EVENTS                     : str       = ("")
    C_RETIRE_NAME                       : str       = "RETIRE"
    C_RETIRE_DESCRIPTION                : str       = ("R Description")

    # memory_bound.py
    C_MEMORY_BOUND_METRICS              : str       = ("stall_memory_dependency,stall_constant_memory_dependency,stall_memory_throttle")
    C_MEMORY_BOUND_EVENTS               : str       = ("")
    C_MEMORY_BOUND_NAME                 : str       = "BACK_END.MEMORY_BOUND"
    C_MEMORY_BOUND_DESCRIPTION          : str       = ("It analyzes the parts of the GPU architecture where we have a loss of performance (IPC) due to\n"
                                                          + "memory bounds. This part takes into account aspects such as data dependencies, failures or access\n"
                                                          + "limits in caches")

    # core_bound.py
    C_CORE_BOUND_METRICS                : str       = ("stall_pipe_busy")
    C_CORE_BOUND_EVENTS                 : str       = ("")
    C_CORE_BOUND_NAME                   : str       = "BACK_END.CORE_BOUND"
    C_CORE_BOUND_DESCRIPTION            : str       = ("In this part, the aspects related to CUDA cores that cause bottlenecks and thus performance losses are analyzed.\n"
                                                        + "Some aspects such as the use and availability of the functional units are analyzed.")

    # front_band_width.py
    C_FRONT_BAND_WIDTH_METRICS          : str       = ("stall_exec_dependency,stall_not_selected")
    C_FRONT_BAND_WIDTH_EVENTS           : str       = ("")
    C_FRONT_BAND_WIDTH_NAME             : str       = "FRONT_END.BAND_WIDTH"
    C_FRONT_BAND_WIDTH_DESCRIPTION      : str       = ("BW description")

    # front_dependency.py
    C_FRONT_DEPENDENCY_METRICS          : str       = ("stall_inst_fetch,stall_sync,stall_other")
    C_FRONT_DEPENDENCY_EVENTS           : str       = ("")
    C_FRONT_DEPENDENCY_NAME             : str       = "FRONT_END.DEPENDENCY"
    C_FRONT_DEPENDENCY_DESCRIPTION      : str       = ("D description")

    # constant_memory_bound.py
    C_CONSTANT_MEMORY_BOUND_METRICS     : str       = ("stall_constant_memory_dependency")
    C_CONSTANT_MEMORY_BOUND_EVENTS      : str       = ("")
    C_CONSTANT_MEMORY_BOUND_NAME        : str       = "MEMORY_BOUND.CONSTANT_MEMORY_BOUND"
    C_CONSTANT_MEMORY_BOUND_DESCRIPTION : str       = ("CMB description")