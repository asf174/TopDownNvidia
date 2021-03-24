"""
Class with all params of BackEnd-MemoryBound class
and their subclasses

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class BackMemoryBoundParameters:

    C_BACK_END_NAME                    : str        = "BACK-END.CORE-BOUND"
    C_BACK_END_DESCRIPTION             : str        = ("In this part, the aspects related to CUDA cores that cause bottlenecks and thus performance losses are analyzed.\n"
                                                        + "Some aspects such as the use and availability of the functional units are analyzed.")
    # back_end_nvprof.py
    C_BACK_END_NVPROF_METRICS          : str        = ("stall_pipe_busy")
    C_BACK_END_NVPROF_EVENTS           : str        = ("")

    # back_end_nsight.py
    C_BACK_END_NSIGHT_METRICS          : str        = ("")
