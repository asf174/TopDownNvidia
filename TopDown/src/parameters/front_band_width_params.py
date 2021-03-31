"""
Class with all params of FrontEnd-BandWidth class
and their subclasses

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class FrontBandWidthParameters:

    C_FRONT_BAND_WIDTH_NAME                    : str       = "FRONT-END.BANDWITH"
    C_FRONT_BAND_WIDTH_DESCRIPTION             : str       = ("F.BW D")
    
    # front_band_width_nvprof.py
    C_FRONT_BAND_WIDTH_NVPROF_METRICS          : str       = ("stall_exec_dependency,stall_not_selected")
    C_FRONT_BAND_WIDTH_NVPROF_EVENTS           : str       = ("")

    # front_band_width_nsight.py
    C_FRONT_BAND_WIDTH_NSIGHT_METRICS          : str       = ("smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct," +
                                                             "smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct," +
                                                            "smsp__warp_issue_stalled_sleeping_per_warp_active.pct")
