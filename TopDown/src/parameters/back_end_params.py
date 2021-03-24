"""
Class with all params of BackEnd class
and their subclasses

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class BackEndParameters:

    C_BACK_END_NAME                    : str        = "BACK-END"
    C_BACK_END_DESCRIPTION             : str        = ("It analyzes the parts of the GPU architecture where the BackEnd produces bottleneck,\n"
                                                    + "which leads to IPC losses. In this part, We analyze aspects related to the 'execution' part of\n"
                                                    + "the instructions, in which aspects such as limitations by functional units, memory limits, etc.\n")
    # back_end_nvprof.py
    C_BACK_END_NVPROF_METRICS          : str        = ("stall_memory_dependency,stall_constant_memory_dependency,stall_pipe_busy," +
                                                        "stall_memory_throttle")
    C_BACK_END_NVPROF_EVENTS           : str        = ("")

    # back_end_nsight.py
    C_BACK_END_NSIGHT_METRICS          : str        = ("")