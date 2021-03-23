"""
Class with all params of FrontEnd class
and their subclasses

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class FrontEndParameters:

    C_FRONT_END_NAME                    : str       = "FRONT-END"
    C_FRONT_END_DESCRIPTION             : str       = ("It analyzes the parts of the GPU architecture where the FrontEnd produces bottlenecks,\n" 
                                                    + "which leads to IPC losses. In this part, aspects related to the fetch of instructions\n"
                                                    + "are analyzed, such as errors in the instruction cache or IPC losses due to thread synchronization.\n")
    # frond_end_nvprof.py
    C_FRONT_END_NVPROF_METRICS          : str       = ("stall_inst_fetch,stall_exec_dependency,stall_sync,stall_other," +
                                                        "stall_not_selected,stall_not_selected")
    C_FRONT_END_NVPROF_EVENTS           : str       = ("")

    # frond_end_nsight.py
    C_FRONT_END_NSIGHT_METRICS          : str       = ("")
    C_FRONT_END_NSIGHT_EVENTS           : str       = ("")