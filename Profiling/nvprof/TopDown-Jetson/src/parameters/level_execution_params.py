"""
Class with all params of LevelExecution file
and their subclasses of the hierachy

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class LevelExecutionParameters:

    C_IPC_METRIC_NAME                   : str       = "ipc"
    C_WARP_EXECUTION_EFFICIENCY_NAME    : str       = "warp_execution_efficiency"
    C_ISSUE_IPC_NAME                    : str       = "issued_ipc"
    C_INFO_MESSAGE_EXECUTION_NVPROF     : str       = "Making analysis... Wait to results."
    C_CYCLES_ELAPSED_NAME               : str       = "elapsed_cycles_sm"
    C_MAX_NUM_RESULTS_DECIMALS          : int       = 3 # recommended be same with same value definided in TopDownParameters