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
    
    # NVPROF metrics/arguments
    C_FRONT_BAND_WIDTH_NVPROF_L2_METRICS          : str       = ("stall_exec_dependency,stall_not_selected")
    C_FRONT_BAND_WIDTH_NVPROF_L2_EVENTS           : str       = ("")

    C_FRONT_BAND_WIDTH_NVPROF_L3_METRICS          : str       = ("stall_exec_dependency,stall_not_selected")
    C_FRONT_BAND_WIDTH_NVPROF_L3_EVENTS           : str       = ("")
    
    # NSIGHT metrics
    C_FRONT_BAND_WIDTH_NSIGHT_L2_METRICS          : str       = ("smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct," +
                                                                 "smsp__warp_issue_stalled_misc_per_warp_active.pct")
    C_FRONT_BAND_WIDTH_NSIGHT_L3_METRICS          : str       = ("smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct," +  
                                                                 "smsp__warp_issue_stalled_misc_per_warp_active.pct")

