"""
Class with all params of Divergence class
and their subclasses

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class DivergenceEndParameters:

    C_DIVERGENCE_NAME                    : str      = "DIVERGENCE"
    C_DIVERGENCE_DESCRIPTION             : str      = ("It analyzes the parts of the GPU architecture where divergence causes a loss of performance.\n" + 
                                                        "This problem is caused when warps are not used correctly. This is caused for example when, for example, there are\n" + 
                                                        "threads (of the same warp) that have to execute an instruction and others do not. In this case there are GPU cores\n" + 
                                                        "that are not being used. Another worst case occurs when some threads of a warp have to execute one instruction and\n" + 
                                                        "others another (if-else). In this case, twice as many cycles are necessary to execute, and in all cases part of the\n" +  
                                                        "cores will not be used.") # TODO preguntar separaciones, si anhadir el espacio
    # divergence_nvprof.py
    C_DIVERGENCE_NVPROF_METRICS          : str      = ("branch_efficiency,warp_execution_efficiency,issued_ipc")
    C_DIVERGENCE_NVPROF_EVENTS           : str      = ("branch,divergent_branch")

    # divergence_nsight.py
    C_DIVERGENCE_NSIGHT_METRICS          : str      = ("")