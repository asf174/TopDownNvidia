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
    C_FRONT_DEPENDENCY_NSIGHT_METRICS          : str       = ("smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct," +
                                                             "smsp__warp_issue_stalled_membar_per_warp_active.pct, smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct," +
                                                             "smsp__warp_issue_stalled_misc_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct")
