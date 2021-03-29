"""
Class with all params of LevelExecution file
and their subclasses of the hierachy

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class LevelExecutionParameters:

    C_IPC_METRIC_NAME_NVPROF                            : str       = "ipc"
    C_IPC_METRIC_NAME_NSIGHT                            : str       = "smsp__inst_executed.avg.per_cycle_active"

    C_WARP_EXECUTION_EFFICIENCY_METRIC_NAME_NVPROF      : str       = "warp_execution_efficiency"
    C_WARP_EXECUTION_EFFICIENCY_METRIC_NAME_NSIGHT      : str       = "smsp__thread_inst_executed_per_inst_executed.ratio"
    
    C_ISSUE_IPC_METRIC_NAME_NVPROF                      : str       = "issued_ipc"
    C_ISSUE_IPC_METRIC_NAME_NSIGHT                      : str       = "smsp__inst_issued.avg.per_cycle_active"

    
    C_CYCLES_ELAPSED_EVENT_NAME_NVPROF                  : str       = "elapsed_cycles_sm"
    C_CYCLES_ELAPSED_METRIC_NAME_NSIGHT                 : str       = "sm__cycles_elapsed.sum"

    C_MAX_NUM_RESULTS_DECIMALS                          : int       = 3 # recommended be same with same value definided in TopDownParameters

    C_INFO_MESSAGE_EXECUTION_NVPROF                     : str       = "Making analysis... Wait to results."

    # If your GPU is not Tesla model,
    # add here the events and metrics 
    # that will be computed by adding in 
    # each kernel, and not as a function 
    # of the time executed
    C_METRICS_AND_EVENTS_NOT_AVERAGE_COMPUTED   : str       = "inst_issued"