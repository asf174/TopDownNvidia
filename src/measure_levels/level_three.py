"""
Class that represents the level three of the execution.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import re
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from measure_levels.level_two import LevelTwo
from measure_parts.memory_constant_memory_bound import MemoryConstantMemoryBound
from errors.level_execution_errors import *
from abc import abstractmethod # abstract class
from pathlib import Path

class LevelThree(LevelTwo):
    """
    Class with level three of the execution.
    """

    @abstractmethod
    def memory_constant_memory_bound(self) -> MemoryConstantMemoryBound:
        """
        Return MemoryConstantMemoryBound part of the execution.

        Returns:
            reference to ConstantMemoryBound part of the execution
        """

        pass

    @abstractmethod
    def _generate_command(self) -> str:
        """ 
        Generate command of execution with NVIDIA scan tool.

        Returns:
            String with command to be executed
        """
        
        pass


    def set_results(self,output_command : str):
        """
        Set results of execution ALREADY DONE. Results are in the argument.

        Params:
            output_command : str    ; str with results of execution.
        """

        super()._set_front_back_divergence_retire_results(output_command) # level one results
        super()._set_memory_core_decode_fetch_results(output_command) # level two
        self._set_memory_constant_memory_bound_results(output_command) # level three
        pass


    @abstractmethod
    def _metricExists(self, metric_name : str) -> bool:
        """
        Check if metric exists in some part of the execution (MemoryBound, CoreBound...). 

        Params:
            metric_name  : str   ; name of the metric to be checked

        Returns:
            True if metric is defined in some part of the execution (MemoryBound, CoreBound...)
            or false in other case
        """

        pass

    def _set_memory_constant_memory_bound_results(self, results_launch : str):
        """
        Set results of the level thre part (that are not level one or two).
        
        Params:
            results_launch : str   ; results generated by NVIDIA scan tool.
        """

        pass 

    def memory_constant_memory_bound_stall(self) -> float:
        """
        Returns percent of stalls due to BackEnd.MemoryBound.MemoryConstantMemoryBound part.

        Returns:
            Float with percent of total stalls due to BackEnd.MemoryBound.Constant_Memory_Bound part
        """

        return self._get_stalls_of_part(self.memory_constant_memory_bound().metrics())
        pass
    
    def memory_constant_memory_bound_stall_on_back(self) -> float:
        """ 
        Obtain the percentage of stalls due to BackEnd.MemoryBound.MemoryConstantMemoryBound.
        on the total BackEnd.

        Returns:
            Float the percentage of stalls due to BackEnd.Memory_Bound
            on the total BackEnd
        """

        return (self.memory_constant_memory_bound_stall()/super().back_end_stall())*100.0

    def memory_constant_memory_bound_stall_on_memory_bound(self) -> float:
        """ 
        Obtain the percentage of stalls due to BackEnd.MemoryBound.MemoryConstantMemoryBound
        on the total BackEnd.MemoryBound.

        Returns:
            Float the percentage of stalls due to BackEnd.MemoryBound.MemoryConstantMemoryBound
            on the total BackEnd.MemoryBound
        """

        return (self.memory_constant_memory_bound_stall()/super().back_memory_bound_stall())*100.0
        pass

    def memory_constant_memory_bound_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to BackEnd.MemoryBound.MemoryConstantMemoryBound part.

        Returns:
            Float with the percent of BackEnd.MemoryBound.MemoryConstantMemoryBound's IPC degradation
        """

        return (((self._stall_ipc()*(self.memory_constant_memory_bound_stall()/100.0))/self.get_device_max_ipc())*100.0)
        pass
