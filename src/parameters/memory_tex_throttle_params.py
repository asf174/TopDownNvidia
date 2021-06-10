"""
Class with all params of BackEnd.MemoryBound.TexThrottle class
and their subclasses

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class MemoryTexThrottleParameters:

    C_MEMORY_TEX_THROTTLE_NAME                    : str        = "BACK-END.MEMORY-BOUND.TEX-THROTTLE"
    C_MEMORY_TEX_THROTTLE_DESCRIPTION             : str        = ("")
    
    # NSIGHT metrics
    C_MEMORY_TEX_THROTTLE_NSIGHT_METRICS          : str        = ("smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct")
