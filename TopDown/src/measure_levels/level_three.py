"""
Class that represents the level three of the execution

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
from shell.shell import Shell # launch shell arguments
from errors.level_execution_errors import *
from show_messages.message_format import MessageFormat
from abc import ABC, abstractmethod # abstract class

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
        
        command : str = ("sudo $(which nvprof) --metrics " + self._front_end.metrics_str() + 
            "," + self._back_end.metrics_str() + "," + self._divergence.metrics_str() + "," +
            self._extra_measure.metrics_str() + "," + self._retire.metrics_str() + "," +
            self._back_core_bound.metrics_str() + "," + self._back_memory_bound.metrics_str() +
            "," + self.__constant_memory_bound.metrics_str() + "  --events " + self._front_end.events_str() +
            "," + self._back_end.events_str() + "," + self._divergence.events_str() +  "," + self._extra_measure.events_str() +
            "," + self._retire.events_str() + "," + self._back_core_bound.events_str() + "," + 
            self._back_memory_bound.events_str() +  "," + self.__constant_memory_bound.events_str() + 
            " --unified-memory-profiling off --profile-from-start off " + self._program)
        return command
        pass

    def run(self, lst_output : list):
        """ 
        Makes execution.
        
        Parameters:
            lst_output  : list ; list with results
        """

        # compute results
        output_command : str = super()._launch(self._generate_command())
        super()._set_front_back_divergence_retire_results(output_command) # level one results
        super()._set_memory_core_bandwith_dependency_results(output_command) # level two
        self._set_constant_memory_bound_results(output_command) # level three
        self._get_results(lst_output)
        pass

    def _metricExists(self, metric_name : str) -> bool:
        """
        Check if metric exists in some part of the execution (MemoryBound, CoreBound...). 

        Params:
            metric_name  : str   ; name of the metric to be checked

        Returns:
            True if metric is defined in some part of the execution (MemoryBound, CoreBound...)
            or false in other case
        """
        
        return True
        pass

    @abstractmethod
    def _set_constant_memory_bound_results(self, results_launch : str):
        """
        Set results of the level thre part (that are not level one or two).
        
        Params:
            results_launch : str   ; results generated by NVIDIA scan tool.
        """

        pass 

    def get_constant_memory_bound_stall(self) -> float:
        """
        Returns percent of stalls due to BackEnd.MemoryBound.Constant_Memory_Bound part.

        Returns:
            Float with percent of total stalls due to BackEnd.MemoryBound.Constant_Memory_Bound part
        """

        return self._get_stalls_of_part(self.memory_constant_memory_bound().metrics())
        pass
    
    def get_constant_memory_bound_stall_on_back(self) -> float:
        """ 
        Obtain the percentage of stalls due to BackEnd.MemoryBound.ConstantMemoryBound # repasar estos nombres en todo
        on the total BackEnd

        Returns:
            Float the percentage of stalls due to BackEnd.Memory_Bound
            on the total BackEnd
        """

        return (self.get_constant_memory_bound_stall()/super().get_back_end_stall())*100.0

    def get_constant_memory_bound_stall_on_memory_bound(self) -> float:
        """ 
        Obtain the percentage of stalls due to BackEnd.MemoryBound.ConstantMemoryBound
        on the total BackEnd.MemoryBound

        Returns:
            Float the percentage of stalls due to BackEnd.MemoryBound.ConstantMemoryBound
            on the total BackEnd.MemoryBound
        """

        return (self.get_constant_memory_bound_stall()/super().get_back_memory_bound_stall())*100.0
        pass

    def constant_memory_bound_percentage_ipc_degradation(self) -> float: # repasar nombres... Incluyen superior TODO
        """
        Find percentage of IPC degradation due to FrontEnd.Dependency part.

        Returns:
            Float with the percent of FrontEnd.Dependency's IPC degradation
        """

        return (((self._stall_ipc()*(self.get_constant_memory_bound_stall()/100.0))/self.get_device_max_ipc())*100.0)
        pass
