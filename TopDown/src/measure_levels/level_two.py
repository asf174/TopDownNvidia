"""
Class that represents the level two of the execution

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import re
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from errors.level_execution_errors import *
from measure_parts.back_core_bound import BackCoreBound
from measure_parts.back_memory_bound import BackMemoryBound
from errors.level_execution_errors import *
from measure_parts.front_band_width import FrontBandWidth
from measure_parts.front_dependency import FrontDependency
from measure_levels.level_one import LevelOne
from show_messages.message_format import MessageFormat
from abc import ABC, abstractmethod # abstract class



class LevelTwo(LevelOne, ABC):
    """
    Class with level two of the execution.
    """

    @abstractmethod
    def back_core_bound(self) -> BackCoreBound:
        """
        Return CoreBound part of the execution.

        Returns:
            reference to CoreBound part of the execution
        """
        
        pass

    @abstractmethod
    def back_memory_bound(self) -> BackMemoryBound:
        """
        Return MemoryBound part of the execution.

        Returns:
            reference to MemoryBound part of the execution
        """
        
        pass

    @abstractmethod
    def front_band_width(self) -> FrontBandWidth:
        """
        Return FrontBandWidth part of the execution.

        Returns:
            reference to FrontBandWidth part of the execution
        """
        
        pass

    @abstractmethod
    def front_dependency(self) -> FrontDependency:
        """
        Return FrontDependency part of the execution.

        Returns:
            reference to FrontDependency part of the execution
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

    def __metricExists(self, metric_name : str) -> bool:
        """
        Check if metric exists in some part of the execution (MemoryBound, CoreBound...). 

        Params:
            metric_name  : str   ; name of the metric to be checked

        Returns:
            True if metric is defined in some part of the execution (MemoryBound, CoreBound...)
            or false in other case
        """

        # revisar si las superiores (core bound sobre back end) tiene que estar definida
        is_defined_front_end_value : str = self.front_end().get_metric_value(metric_name)
        is_defined_front_end_description : str = self.front_end().get_metric_description(metric_name)

        is_defined_back_end_value : str = self.back_end().get_metric_value(metric_name)
        is_defined_back_end_description : str = self.back_end().get_metric_description(metric_name)

        is_defined_divergence_value : str = self.divergence().get_metric_value(metric_name)
        is_defined_divergence_description : str = self.divergence().get_metric_description(metric_name)

        is_defined_retire_value : str = self.retire().get_metric_value(metric_name)
        is_defined_retire_description : str = self.retire().get_metric_description(metric_name)

        is_defined_extra_measure_value : str = self.extra_measure().get_metric_value(metric_name)
        is_defined_extra_measure_description : str = self.extra_measure().get_metric_description(metric_name)

        # comprobar el "if not"
        if not ((is_defined_front_end_value is None and is_defined_front_end_description is None) 
            or (is_defined_back_end_value is None and is_defined_back_end_description is None)
            or (is_defined_divergence_value is None and is_defined_divergence_description is None) 
            or (is_defined_retire_value is None or is_defined_retire_description is None)
            or (is_defined_extra_measure_value is None and is_defined_extra_measure_description is None)):
            return False
        return True
        pass

    def __eventExists(self, event_name : str) -> bool:
        """
        Check if event exists in some part of the execution (MemoryBound, CoreBound...). 

        Params:
            event_name  : str   ; name of the event to be checked

        Returns:
            True if event is defined in some part of the execution (MemoryBound, CoreBound...)
            or false in other case
        """

        # revisar si las superiores (core bound sobre back end pe) tiene que estar definida
        is_defined_front_end_value : str = self.front_end().get_metric_value(event_name)
        is_defined_front_end_description : str = self.front_end().get_metric_description(event_name)

        is_defined_back_end_value : str = self.back_end().get_metric_value(event_name)
        is_defined_back_end_description : str = self.back_end().get_metric_description(event_name)

        is_defined_divergence_value : str = self.divergence().get_metric_value(event_name)
        is_defined_divergence_description : str = self.divergence().get_metric_description(event_name)

        is_defined_retire_value : str = self.retire().get_metric_value(event_name)
        is_defined_retire_description : str = self.retire().get_metric_description(event_name)

        is_defined_extra_measure_value : str = self.extra_measure().get_metric_value(event_name)
        is_defined_extra_measure_description : str = self.extra_measure().get_metric_description(event_name)

        # revisar IF, puede ser (1 y 0) y (1 y 1). EL primero no puede darse pero cumpliria
        if not ((is_defined_front_end_value is None and is_defined_front_end_description is None) 
            or (is_defined_back_end_value is None and is_defined_back_end_description is None)
            or (is_defined_divergence_value is None and is_defined_divergence_description is None) 
            or (is_defined_retire_value is None or is_defined_retire_description is None)
            or (is_defined_extra_measure_value is None and is_defined_extra_measure_description is None)):
            return False
        return True
        pass

    @abstractmethod
    def _get_results(self, lst_output : list):
        """ 
        Get results of the different parts.

        Parameters:
            lst_output              : list     ; OUTPUT list with results
        """
        
        pass

    def _set_memory_core_bandwith_dependency_results(self, results_launch : str):
        """
        Set results of the level two part (that are not level one).
        
        Params:
            results_launch : str   ; results generated by NVIDIA scan tool.
        """
        
        pass

    def run(self, lst_output : list):
        """Run execution."""
        
        # compute results
        output_command : str = super()._launch(self._generate_command())
        super()._set_front_back_divergence_retire_results(output_command) # level one results
        self._set_memory_core_bandwith_dependency_results(output_command)
        self._get_results(lst_output)
        pass
    
    def back_core_bound_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to BackEnd.Core_Bound part.

        Returns:
            Float with the percent of BackEnd.Core_Bound's IPC degradation
        """
        
        return (((self._stall_ipc()*(self.get_back_core_bound_stall()/100.0))/super().get_device_max_ipc())*100.0)
        pass

    def back_memory_bound_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to BackEnd.Memory_Bound part.

        Returns:
            Float with the percent of BackEnd.Memory_Bound's IPC degradation
        """

        return (((self._stall_ipc()*(self.get_back_memory_bound_stall()/100.0))/super().get_device_max_ipc())*100.0)
        pass

    def front_band_width_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to FrontEnd.BandWidth part.

        Returns:
            Float with the percent of FrontEnd.BandWidth's IPC degradation
        """

        return (((self._stall_ipc()*(self.get_front_band_width_stall()/100.0))/super().get_device_max_ipc())*100.0)
        pass

    def front_dependency_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to FrontEnd.Dependency part.

        Returns:
            Float with the percent of FrontEnd.Dependency's IPC degradation
        """

        return (((self._stall_ipc()*(self.get_front_dependency_stall()/100.0))/super().get_device_max_ipc())*100.0)
        pass

    def get_back_memory_bound_stall(self) -> float:
        """
        Returns percent of stalls due to BackEnd.Memory_Bound part.

        Returns:
            Float with percent of total stalls due to BackEnd.Memory_Bound
        """

        return self._get_stalls_of_part(self._back_memory_bound.metrics())
        pass

    def get_back_core_bound_stall(self) -> float:
        """
        Returns percent of stalls due to BackEnd.Core_Bound part.

        Returns:
            Float with percent of total stalls due to BackEnd.Core_Bound
        """

        return self._get_stalls_of_part(self._back_core_bound.metrics())
        pass

    def get_front_band_width_stall(self) -> float:
        """
        Returns percent of stalls due to FrontEnd.Band_width part.

        Returns:
            Float with percent of total stalls due to FrontEnd.Band_width part
        """

        return self._get_stalls_of_part(self._front_band_width.metrics())
        pass

    def get_front_dependency_stall(self) -> float:
        """
        Returns percent of stalls due to FrontEnd.Dependency part.

        Returns:
            Float with percent of total stalls due to FrontEnd.Dependency part
        """

        return self._get_stalls_of_part(self._front_dependency.metrics())
        pass

    def get_back_memory_bound_stall_on_back(self) -> float:
        """ 
        Obtain the percentage of stalls due to BackEnd.Memory_Bound
        on the total BackEnd

        Returns:
            Float the percentage of stalls due to BackEnd.Memory_Bound
            on the total BackEnd
        """

        return (self.get_back_memory_bound_stall()/super().get_back_end_stall())*100.0 

    def get_back_core_bound_stall_on_back(self) -> float:
        """ 
        Obtain the percentage of stalls due to BackEnd.Core_Bound
        on the total BackEnd

        Returns:
            Float the percentage of stalls due to BackEnd.Core_Bound
            on the total BackEnd
        """

        return (self.get_back_core_bound_stall()/super().get_back_end_stall())*100.0 

    def get_front_band_width_stall_on_front(self) -> float:
        """ 
        Obtain the percentage of stalls due to FrontEnd.Band_width
        on the total FrontEnd

        Returns:
            Float the percentage of stalls due to FrontEnd.Band_width
            on the total FrontEnd
        """

        return (self.get_front_band_width_stall()/super().get_front_end_stall())*100.0 
        pass

    def get_front_dependency_stall_on_front(self) -> float:
        """ 
        Obtain the percentage of stalls due to FrontEnd.Dependency
        on the total FrontEnd

        Returns:
            Float the percentage of stalls due to FrontEnd.Dependency
            on the total FrontEnd
        """

        return (self.get_front_dependency_stall()/super().get_front_end_stall())*100.0 
        pass
