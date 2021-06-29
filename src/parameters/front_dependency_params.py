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

    # NVPROF metrics/arguments
    C_FRONT_DEPENDENCY_NVPROF_L2_METRICS          : str       = ("stall_inst_fetch,stall_sync,stall_other")
    C_FRONT_DEPENDENCY_NVPROF_L2_EVENTS           : str       = ("")
    
    C_FRONT_DEPENDENCY_NVPROF_L3_METRICS          : str       = ("stall_inst_fetch,stall_sync,stall_other")
    C_FRONT_DEPENDENCY_NVPROF_L3_EVENTS           : str       = ("")
    
    # NSIGHT metrics
    C_FRONT_DEPENDENCY_NSIGHT_L2_METRICS          : str       = ("smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct," +
                                                                "smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct," +
                                                                "smsp__warp_issue_stalled_misc_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct")
    C_FRONT_DEPENDENCY_NSIGHT_L3_METRICS          : str       = ("smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct," +
                                                                "smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct," +
                                                                "smsp__warp_issue_stalled_misc_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct")

