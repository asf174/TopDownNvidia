"""
Class that represents the level one of the execution

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import re
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(1, parentdir) 
from measure_levels.level_execution import LevelExecution
from parameters.level_execution_params import LevelExecutionParameters
from errors.level_execution_errors import *
from measure_parts.front_end import FrontEnd 
from measure_parts.back_end import BackEnd
from measure_parts.divergence import Divergence
from measure_parts.retire import Retire
from show_messages.message_format import MessageFormat
from abc import ABC, abstractmethod # abstract class
from graph.pie_chart import PieChart

class LevelOne(LevelExecution, ABC):
 
    @abstractmethod
    def _generate_command(self) -> str:
        """ 
        Generate command of execution with NVIDIA scan tool.

        Returns:
            String with command to be executed
        """

        pass

    @abstractmethod
    def _get_results(self, lst_output : list):
        """
        Get results of the different parts.

        Parameters:
            lst_output              : list     ; OUTPUT list with results
        """
        
        pass
 
    def set_results(output_command : str):
        """
        Set results of execution ALREADY DONE. Results are in the argument.

        Params:
            output_command : str    ; str with results of execution.
        """
        
        self._set_front_back_divergence_retire_results(output_command)
        pass
    
    def run(self, lst_output : list):
        """Run execution."""
        
        output_command : str = super()._launch(self._generate_command())
        self.set_results(output_command)
        self._get_results(lst_output)
        pass
    
    def _get_ipc(self, ipc_metric_name : str) -> float:
        """
        Get IPC of execution based on metric name.

        Params:
            ipc_metric_name : str ; name of metric computed by NVIDIA scan tool

        Raises:
            IpcMetricNotDefined ; raised if IPC cannot be obtanied because it was not 
            computed by the NVIDIA scan tool.

        Returns:
            float with the IPC
        """

        ipc_list : list = self._retire.get_metric_value(ipc_metric_name)
        if ipc_list is None:
            raise IpcMetricNotDefined # revisar TODO
        total_ipc : float = self._get_total_value_of_list(ipc_list, True)
        return total_ipc
        pass

    @abstractmethod
    def ipc(self) -> float:
        """
        Get IPC of execution based on metric name.

        Returns:
            float with the IPC
        """

        pass

    def _ret_ipc(self, warp_exec_efficiency_name : str) -> float:
        """
        Get "RETIRE" IPC of execution based on warp execution efficiency metric name

        Params:
            warp_exec_efficiency_name : str ; string with warp execution efficiency metricc name

        Raises:
            RetireIpcMetricNotDefined ; raised if retire IPC cannot be obtanied because it was not 
            computed by the NVIDIA scan tool.
        """

        warp_execution_efficiency_list : list = self._divergence.get_metric_value(warp_exec_efficiency_name) #TODO importante
        if warp_execution_efficiency_list is None:
            raise RetireIpcMetricNotDefined
        total_warp_execution_efficiency : float = self._get_total_value_of_list(warp_execution_efficiency_list, True)
        return self.ipc()*(total_warp_execution_efficiency/100.0)
        pass

    @abstractmethod
    def retire_ipc(self) -> float:
        """
        Get "RETIRE" IPC of execution.

        Raises:
            RetireIpcMetricNotDefined ; raised if retire IPC cannot be obtanied because it was not 
            computed by the NVIDIA scan tool.
        """

        pass
    
    @abstractmethod
    def retire_ipc_per_kernel(self) -> float:
        """
        Get "RETIRE" IPC of execution.

        Raises:
            RetireIpcMetricNotDefined ; raised if retire IPC cannot be obtanied because it was not 
            computed by the NVIDIA scan tool.
        """

        pass


    
    @abstractmethod
    def front_end(self) -> FrontEnd:
        """
        Return FrontEnd part of the execution.

        Returns:
            reference to FrontEnd part of the execution
        """

        pass
    
    @abstractmethod
    def back_end(self) -> BackEnd:
        """
        Return BackEnd part of the execution.

        Returns:
            reference to BackEnd part of the execution
        """

        pass

    @abstractmethod
    def divergence(self) -> Divergence:
        """
        Return Divergence part of the execution.

        Returns:
            reference to Divergence part of the execution
        """

        pass

    @abstractmethod
    def retire(self) -> Retire:
        """
        Return Retire part of the execution.

        Returns:
            reference to Retire part of the execution
        """

        pass

    def front_end_stall(self) -> float:
        """
        Returns percent of stalls due to FrontEnd part.

        Returns:
            Float with percent of total stalls due to FrontEnd
        """

        return super()._get_stalls_of_part(self._front_end.metrics())
        pass
    
    def back_end_stall(self) -> float:
        """
        Returns percent of stalls due to BackEnd part.

        Returns:
            Float with percent of total stalls due to BackEnd
        """

        return super()._get_stalls_of_part(self._back_end.metrics())
        pass
    
    def front_end_stall_per_kernel(self) -> list[float]:
        """
        Returns percent of stalls due to FrontEnd part in each kernel.

        Returns:
            Float with percent of total stalls due to FrontEnd
        """

        return super()._get_stalls_of_part_per_kernel(self._front_end.metrics())
        pass
    
    def back_end_stall_per_kernel(self) -> float:
        """
        Returns percent of stalls due to BackEnd part in each kernel.

        Returns:
            Float with percent of total stalls due to BackEnd
        """

        return supe()._get_stalls_of_part_per_kernel(self._back_end.metrics())
        pass


    def _diver_ipc_degradation(self, warp_exec_efficiency_name  : str, issue_ipc_name : str) -> float:
        """
        Find IPC degradation due to Divergence part based on the name of the required metric.

        Params:
            warp_exec_efficiency_name  : str   ; name of metric to obtain warp execution efficiency
            issue_ipc_name             : str   ; name of metric to bain issue ipc
        Returns:
            Float with the Divergence's IPC degradation

        Raises:
            RetireIpcMetricNotDefined ; raised if retire IPC cannot be obtanied because it was not 
            computed by the NVIDIA scan tool.
        """

        ipc : float = self.ipc() 
        warp_execution_efficiency_list  : list = self._divergence.get_metric_value(warp_exec_efficiency_name)
        if warp_execution_efficiency_list is None:
            raise RetireIpcMetricNotDefined # revisar si crear otra excepcion (creo que si)
        total_warp_execution_efficiency : float = self._get_total_value_of_list(warp_execution_efficiency_list, True)
        issued_ipc_list : list = self._divergence.get_metric_value(issue_ipc_name)
        total_issued_ipc : float = self._get_total_value_of_list(issued_ipc_list, True)
        ipc_diference : float = float(total_issued_ipc) - ipc
        if ipc_diference < 0.0:
            ipc_diference = 0.0
        return ipc * (1.0 - (total_warp_execution_efficiency/100.0)) + ipc_diference
        pass

    @abstractmethod
    def _divergence_ipc_degradation(self) -> float:
        """
        Find IPC degradation due to Divergence part

        Returns:
            Float with theDivergence's IPC degradation

        """

        pass

    def _stall_ipc(self) -> float:
        """
        Find IPC due to STALLS

        Returns:
            Float with STALLS' IPC degradation
        """

        return super().get_device_max_ipc() - self.retire_ipc() - self._divergence_ipc_degradation()
        pass

    def _stall_ipc_per_kernel(self) -> list[float]:
        """
        Find IPC due to STALLS

        Returns:
            Float with STALLS' IPC degradation
        """
        
        max_ipc : float = super().get_device_max_ipc()
        list_retire_ipc : list[float] = self.retire_ipc_per_kernel()
        list_divergence_ipc_degradation : list[float] = self._divergence_ipc_degradation_per_kernel()
        list_stall_ipc_per_kernel : list[float] = list()
        for i in range(0, len(super().kernels())):
            list_stall_ipc_per_kernel.append(max_ipc - list_retire_ipc[i] - list_divergence_ipc_degradation[i])
        return list_stall_ipc_per_kernel
        pass


    def divergence_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to Divergence part.

        Returns:
            Float with the percent of Divergence's IPC degradation
        """

        return (self._divergence_ipc_degradation()/super().get_device_max_ipc())*100.0
        pass

    def front_end_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to FrontEnd part.

        Returns:
            Float with the percent of FrontEnd's IPC degradation
        """
        
        return ((self._stall_ipc()*(self.front_end_stall()/100.0))/self.get_device_max_ipc())*100.0
        pass

    def back_end_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to BackEnd part.

        Returns:
            Float with the percent of BackEnd's IPC degradation
        """
        
        return ((self._stall_ipc()*(self.back_end_stall()/100.0))/super().get_device_max_ipc())*100.0
        pass

    def retire_ipc_percentage(self) -> float:
        """
        Get percentage of TOTAL IPC due to RETIRE.

        Returns:
            Float with percentage of TOTAL IPC due to RETIRE
        """
        return (self.ipc()/super().get_device_max_ipc())*100.0
 
    def divergence_percentage_ipc_degradation_per_kernel(self) -> list[float]:
        """
        Find percentage of IPC degradation due to Divergence part in each kernel.

        Returns:
            Float with the percent of Divergence's IPC degradation
        """
        
        max_ipc : float = supercentage().get_device_max_ipc()
        list_divergence_ipc_degradation : list[float] = self._divergence_ipc_degradation_percentage_kernel()
        list_percentage_ipc_degradation_percentage_kernel : list[float] = list()
        for i in range(0, len(supercentage().kernels())):
         list_percentage_ipc_degradation_percentage_kernel.append((list_divergence_ipc_degradation[i]/max_ipc)*100.0)
        return list_percentage_ipc_degradation_percentage_kernel
        pass

    def front_end_percentage_ipc_degradation_per_kernel(self) -> list[float]:
        """
        Find percentage of IPC degradation due to FrontEnd part.

        Returns:
            Float with the percent of FrontEnd's IPC degradation
        """
        
        return ((self._stall_ipc()*(self.front_end_stall()/100.0))/self.get_device_max_ipc())*100.0
        pass

    def back_end_percentage_ipc_degradation_per_kernel(self) -> list[float]:
        """
        Find percentage of IPC degradation due to BackEnd part.

        Returns:
            Float with the percent of BackEnd's IPC degradation
        """
        
        max_ipc : float = super().get_device_max_ipc()
        list_stall_ipc_per_kernel : list[float] = self._stall_ipc_per_kernel()
        list_back_end_stall_per_kernel : list[float] = self.back_end_stall_per_kernel()
        list_back_end_percentage_ipc_degra_per_kernel : list[float] = list()
        for i in range(0, len(super().kernels())):
            list_back_end_percentage_ipc_degra_per_kernel.append((list_stall_ipc_per_kernel[i]*(list_back_end_stall_per_kernel[i]/100.0))/max_ipc)*100.0
        return list_back_end_percentage_ipc_degra_per_kernel
        pass

    def retire_ipc_percentage_per_kernel(self) -> list[float]:
        """
        Get percentage of TOTAL IPC due to RETIRE.

        Returns:
            Float with percentage of TOTAL IPC due to RETIRE
        """

        list_ipc_per_kernel : list[float] = self.ipc_per_kernel()
        max_ipc : float = super().get_device_max_ipc()
        retire_ipc_percentage_per_kernel : list[float] = list()
        for i in range(0, len(super().kernels())):
            retire_ipc_percentage_per_kernel.append((list_ipc_per_kernel[i] / max_ipc)*100.0)
        return retire_ipc_percentage_per_kernel 


    @abstractmethod
    def _set_front_back_divergence_retire_results(self, results_launch : str):
        """ Get Results from FrontEnd, BanckEnd, Divergence and Retire parts.
        
        Params:
            results_launch  : str   ; results generated by NVIDIA scan tool
            
        Raises:
            EventNotAsignedToPart       ; raised when an event has not been assigned to any analysis part * (NVPROF mode only)
            MetricNotAsignedToPart      ; raised when a metric has not been assigned to any analysis part 
        """
        
        pass

    def _create_graph(self) -> PieChart:
        """ 
        Create a graph where figures are going to be saved.

        Returns:
            Referente to PieChart with graph
        """

        titles_graphs : list = LevelExecutionParameters.C_LEVEL_ONE_GRAPHS_TITLES
        if len(titles_graphs) < 2:
            raise GraphsTitleSizeError
        return PieChart(1,2, "Description of Results", titles_graphs) # pie chart graph
        pass
   
    def _add_graph_data(self, graph : PieChart):
        """ 
        Add data to graph.

        Params:
            graph   : PieChart  ; reference to PieChart where save figures
            title   : str       ; title of graph to be added        
        """

        labels : list = [self._front_end.name(), self._back_end.name(), self._divergence.name(), self._retire.name()]
        values : list = [self.front_end_percentage_ipc_degradation(), self.back_end_percentage_ipc_degradation(), 
            self.divergence_percentage_ipc_degradation(), self.retire_ipc()]
        graph.add_graph(labels, values, "1")
        values = [self.front_end_stall(), self.back_end_stall()]
        graph.add_graph(labels, values, "1")
        
        pass
