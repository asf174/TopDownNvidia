"""
Class with all params of topdown.py file

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class Parameters:
  
    """
    Options of program
    """
    # Level
    C_LEVEL_SHORT_OPTION                : str       = "-l"
    C_LEVEL_LONG_OPTION                 : str       = "--level"
    
    # Long description
    C_LONG_DESCRIPTION_SHORT_OPTION     : str       = "-ld"
    C_LONG_DESCRIPTION_LONG_OPTION      : str       = "--long-desc"
    
    # Help 
    C_HELP_SHORT_OPTION                 : str       = "-h"
    C_HELP_LONG_OPTION                  : str       = "--help"

    # Output file
    C_OUTPUT_FILE_SHORT_OPTION          : str       = "-o"
    C_OUTPUT_FILE_LONG_OPTION           : str       = "--output"

    # Program file
    C_FILE_SHORT_OPTION                 : str       = "-f"
    C_FILE_LONG_OPTION                  : str       = "--file"

    """Options."""
    C_MAX_LEVEL_EXECUTION               : int       = 2
    C_MIN_LEVEL_EXECUTION               : int       = 1
    
    """Metrics."""
    C_LEVEL_1_FRONT_END_METRICS         : str       = ("stall_inst_fetch,stall_exec_dependency,stall_sync,stall_other," +
                                                        "stall_not_selected,stall_not_selected")
    C_LEVEL_1_BACK_END_METRICS          : str       = ("stall_memory_dependency,stall_constant_memory_dependency,stall_pipe_busy," +
                                                        "stall_memory_throttle")
    C_LEVEL_1_DIVERGENCE_METRICS        : str       = ("branch_efficiency,warp_execution_efficiency,warp_nonpred_execution_efficiency")
    
    C_INFO_MESSAGE_EXECUTION_NVPROF     : str       = "Launching Command... Wait to results."