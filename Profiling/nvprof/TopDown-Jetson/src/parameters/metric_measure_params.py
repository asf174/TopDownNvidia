"""
Class with all params of MetricMetricMeasure class
and their subclasses

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class MetricMeasureParameters:

    # metric_base.py

    # front_end.py
    C_FRONT_END_METRICS             : str       = ("stall_inst_fetch,stall_exec_dependency,stall_sync,stall_other," +
                                                        "stall_not_selected,stall_not_selected")
    C_FRONT_END_EVENTS              : str       = ("")

    C_FRONT_END_NAME                : str       = "FRONT-END"
    C_FRONT_END_DESCRIPTION         : str       = ("It analyzes the parts of the GPU architecture where the FrontEnd produces bottlenecks, \n" 
                                                    + "which leads to IPC losses. In this part, aspects related to the fetch of instructions\n"
                                                    + "are analyzed, such as errors in the instruction cache or IPC losses due to thread synchronization.\n")

    # back_end.py
    C_BACK_END_METRICS              : str       = ("stall_memory_dependency,stall_constant_memory_dependency,stall_pipe_busy," +
                                                        "stall_memory_throttle")
    C_BACK_END_EVENTS               : str       = ("")
    C_BACK_END_NAME                 : str       = "BACK-END"
    C_BACK_END_DESCRIPTION          : str       = ("It analyzes the parts of the GPU architecture where the BackEnd produces bottleneck, \n"
                                                    + "which leads to IPC losses. In this part, We analyze aspects related to the 'execution' part of \n"
                                                    + "the instructions, in which aspects such as limitations by functional units, memory limits, etc. \n")

    # divergence.py
    C_DIVERGENCE_METRICS            : str       = ("branch_efficiency,warp_execution_efficiency,issued_ipc")
    C_DIVERGENCE_EVENTS             : str       = ("branch,divergent_branch")
    C_DIVERGENCE_NAME               : str       = "DIVERGENCE"
    C_DIVERGENCE_DESCRIPTION        : str       = ("It analyzes the parts of the GPU architecture where the where divergence causes a loss of performance. \n" + 
                                                    "This problem is caused when warps are not used correctly. This is caused for example when, for example, there are \n" + 
                                                    "threads (of the same warp) that have to execute an instruction and others do not. In this case there are GPU cores \n" + 
                                                    "that are not being used. Another worst case occurs when some threads of a warp have to execute one instruction and \n" + 
                                                    "others another (if-else). In this case, twice as many cycles are necessary to execute, and in all cases part of the \n" +  
                                                    "cores will not be used.") # TODO preguntar separaciones, si anhadir el espacio

    # extra_measure.py
    C_EXTRA_MEASURE_METRICS         : str       = ("inst_issued")
    C_EXTRA_MEASURE_EVENTS          : str       = ("active_cycles")
    C_EXTRA_MEASURE_NAME            : str       = "EXTRA_MEASURE"
    C_EXTRA_MEASURE_DESCRIPTION     : str       = ("EM Description") # TODO preguntar separaciones, si anhadir el espacio

    # retire.py
    C_RETIRE_METRICS                : str       = ("ipc")
    C_RETIRE_EVENTS                 : str       = ("")
    C_RETIRE_NAME                   : str       = "RETIRE"
    C_RETIRE_DESCRIPTION            : str       = ("R Description")
pass

