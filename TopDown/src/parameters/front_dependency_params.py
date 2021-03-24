"""
Class with all params of FrontEnd-Dependency class
and their subclasses

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class FrontDependencyParameters:

    C_FRONT_DEPENDENCY_NAME                    : str       = "FRONT-END.DEPENDENCY"
    C_FRONT_DEPENDENCY_DESCRIPTION             : str       = ("D description")

    # frond_end_nvprof.py
    C_FRONT_DEPENDENCY_NVPROF_METRICS          : str       = ("stall_inst_fetch, stall_sync,stall_other")
    C_FRONT_DEPENDENCY_NVPROF_EVENTS           : str       = ("")

    # frond_end_nsight.py
    C_FRONT_DEPENDENCY_NSIGHT_METRICS          : str       = ("")
