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
from abc import ABC, abstractmethod # abstract class
from graph.pie_chart import PieChart 
from parameters.level_execution_params import LevelExecutionParameters
from measure_parts.divergence_replay import DivergenceReplay
from measure_parts.divergence_branch import DivergenceBranch

class LevelTwo(LevelOne, ABC):
    """
    Class with level two of the execution.
    """
    
     
    @abstractmethod
    def divergence_replay(self) -> DivergenceReplay:
        """
        Return Replay part of the execution.

        Returns:
            reference to CoreBound part of the execution
        """
        
        pass

  
    @abstractmethod
    def divergence_branch(self) -> DivergenceBranch:
        """
        Return Replay part of the execution.

        Returns:
            reference to CoreBound part of the execution
        """
        
        pass


    
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
        Raises:
             MetricNotAsignedToPart ; raised if some metric is found don't assigned 
                                      to any measure part
 
             *EventNotAsignedToPart ; raised if some event is found don't assigned 
                                      to any measure part (only in NVPROF mode)
        """

        pass
    
    def set_results(self,output_command : str):
        """
        Set results of execution ALREADY DONE. Results are in the argument.

        Params:
            output_command : str    ; str with results of execution.
        """

        super()._set_front_back_divergence_retire_results(output_command) # level one results
        self._set_memory_core_bandwith_dependency_results(output_command)
        pass

    def run(self, lst_output : list):
        """Run execution."""
        
        # compute results
        output_command : str = super()._launch(self._generate_command())
        self.set_results(output_command)
        self._get_results(lst_output)
        pass
    
    def back_core_bound_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to BackEnd.Core_Bound part.

        Returns:
            Float with the percent of BackEnd.Core_Bound's IPC degradation
        """
        return (((self._stall_ipc()*(self.back_core_bound_stall()/100.0))/super().get_device_max_ipc())*100.0)
        pass

    def back_memory_bound_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to BackEnd.Memory_Bound part.

        Returns:
            Float with the percent of BackEnd.Memory_Bound's IPC degradation
        """

        return (((self._stall_ipc()*(self.back_memory_bound_stall()/100.0))/super().get_device_max_ipc())*100.0)
        pass

    def front_band_width_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to FrontEnd.BandWidth part.

        Returns:
            Float with the percent of FrontEnd.BandWidth's IPC degradation
        """

        return (((self._stall_ipc()*(self.front_band_width_stall()/100.0))/super().get_device_max_ipc())*100.0)
        pass

    def front_dependency_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to FrontEnd.Dependency part.

        Returns:
            Float with the percent of FrontEnd.Dependency's IPC degradation
        """

        return (((self._stall_ipc()*(self.front_dependency_stall()/100.0))/super().get_device_max_ipc())*100.0)
        pass

    def back_memory_bound_stall(self) -> float:
        """
        Returns percent of stalls due to BackEnd.Memory_Bound part.

        Returns:
            Float with percent of total stalls due to BackEnd.Memory_Bound
        """
        
        return self._get_stalls_of_part(self._back_memory_bound.metrics())
        pass

    def back_core_bound_stall(self) -> float:
        """
        Returns percent of stalls due to BackEnd.Core_Bound part.

        Returns:
            Float with percent of total stalls due to BackEnd.Core_Bound
        """

        return self._get_stalls_of_part(self._back_core_bound.metrics())
        pass

    def front_band_width_stall(self) -> float:
        """
        Returns percent of stalls due to FrontEnd.Band_width part.

        Returns:
            Float with percent of total stalls due to FrontEnd.Band_width part
        """

        return self._get_stalls_of_part(self._front_band_width.metrics())
        pass

    def front_dependency_stall(self) -> float:
        """
        Returns percent of stalls due to FrontEnd.Dependency part.

        Returns:
            Float with percent of total stalls due to FrontEnd.Dependency part
        """

        return self._get_stalls_of_part(self._front_dependency.metrics())
        pass

    def back_memory_bound_stall_on_back(self) -> float:
        """ 
        Obtain the percentage of stalls due to BackEnd.Memory_Bound
        on the total BackEnd

        Returns:
            Float the percentage of stalls due to BackEnd.Memory_Bound
            on the total BackEnd
        """

        return (self.back_memory_bound_stall()/super().back_end_stall())*100.0 

    def back_core_bound_stall_on_back(self) -> float:
        """ 
        Obtain the percentage of stalls due to BackEnd.Core_Bound
        on the total BackEnd

        Returns:
            Float the percentage of stalls due to BackEnd.Core_Bound
            on the total BackEnd
        """

        return (self.back_core_bound_stall()/super().back_end_stall())*100.0 

    def front_band_width_stall_on_front(self) -> float:
        """ 
        Obtain the percentage of stalls due to FrontEnd.Band_width
        on the total FrontEnd

        Returns:
            Float the percentage of stalls due to FrontEnd.Band_width
            on the total FrontEnd
        """

        return (self.front_band_width_stall()/super().front_end_stall())*100.0 
        pass

    def front_dependency_stall_on_front(self) -> float:
        """ 
        Obtain the percentage of stalls due to FrontEnd.Dependency
        on the total FrontEnd

        Returns:
            Float the percentage of stalls due to FrontEnd.Dependency
            on the total FrontEnd
        """

        return (self.front_dependency_stall()/super().front_end_stall())*100.0 
        pass

    @abstractmethod
    def _branch_divergence_ipc_degradation(self) -> float:
        """
        Find IPC degradation due to Divergence.Branch part

        Returns:
            Float with theDivergence's IPC degradation

        """

        pass   
 
    @abstractmethod
    def _replay_divergence_ipc_degradation(self) -> float:
        """
        Find IPC degradation due to Divergence.Replay part

        Returns:
            Float with theDivergence's IPC degradation

        """

        pass   
    
    def _branch_diver_ipc_degradation(self, warp_exec_efficiency_name  : str) -> float:
        """
        Find IPC degradation due to Branch Divergence part based on the name of the required metric.

        Params:
            warp_exec_efficiency_name  : str   ; name of metric to obtain warp execution efficiency
        
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
        return ipc * (1.0 - (total_warp_execution_efficiency/100.0))
        pass

    
    def _replay_diver_ipc_degradation(self, issue_ipc_name : str) -> float:
        """
        Find IPC degradation due to Replay Divergence part based on the name of the required metric.

        Params:
            issued_ipc_name             : str   ; name of metric to obtain issued ipc
        
        Returns:
            Float with the Divergence's IPC degradation

        Raises:
            RetireIpcMetricNotDefined ; raised if retire IPC cannot be obtanied because it was not
            computed by the NVIDIA scan tool.
        """

        ipc : float = self.ipc()
        issued_ipc_list : list = self._divergence.get_metric_value(issue_ipc_name)
        total_issued_ipc : float = self._get_total_value_of_list(issued_ipc_list, True)
        ipc_diference : float = float(total_issued_ipc) - ipc
        if ipc_diference < 0.0:
            ipc_diference = 0.0
        return ipc_diference
        pass

    def branch_divergence_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to Divergence.Branch part.

        Returns:
            Float with the percent of Divergence.Branch's IPC degradation
        """

        return (self._branch_divergence_ipc_degradation()/super().get_device_max_ipc())*100.0
        pass
    
    def replay_divergence_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to Divergence.Replay part.

        Returns:
            Float with the percent of Divergence.Branch's IPC degradation
        """

        return (self._replay_divergence_ipc_degradation()/super().get_device_max_ipc())*100.0
        pass
    

    def _create_graph(self) -> PieChart:
        """ 
        Create a graph where figures are going to be saved.

        Returns:
            Referente to PieChart with graph
        """
        
        titles_graphs : list = LevelExecutionParameters.level_two_graphs_titles(self._front_end.name(), self._back_end.name())
        if len(titles_graphs) < 6:
            raise TitleSizeError
        return PieChart(3, 2, "Description of Results", titles_graphs) # pie chart graph
        pass

    def _add_graph_data(self, graph : PieChart):
        """ 
        Add data to graph.

        Params:
            graph   : PieChart  ; reference to PieChart where save figures        
        """
        
        labels : list = [self._front_end.name(), self._front_band_width.name(), self._front_dependency.name(), self._back_end.name(), self._back_core_bound.name(), 
            self._back_memory_bound.name(), self._divergence.name(), self._retire.name()]
        
        # Level One
        values : list = [super().front_end_percentage_ipc_degradation(), None, None, super().back_end_percentage_ipc_degradation(), None, 
            super().divergence_percentage_ipc_degradation(), super().retire_ipc()]
        
        graph.add_graph(labels, values, "1")
        values = [self.front_end_stall(), None, None, self.back_end_stall(), None, None, None, None]
        graph.add_graph(labels, values, "1")
        
        # Level TWO
        values = [None, self.front_dependency_percentage_ipc_degradation(), self.front_band_width_percentage_ipc_degradation(), None, self.back_core_bound_percentage_ipc_degradation(), 
            self.back_memory_bound_percentage_ipc_degradation(), super().divergence_percentage_ipc_degradation(), super().retire_ipc()] # IPC Degradation
        graph.add_graph(labels, values, "1")
        
        values = [None, self.front_band_width_stall(), self.front_dependency_stall(), None, self.back_core_bound_stall(), self.back_memory_bound_stall(), None, None] # Stalls total
        graph.add_graph(labels, values, "1")
        
        values = [None, self.front_band_width_stall_on_front(), self.front_dependency_stall_on_front(), None, None, None, None, None] # Stalls on FrontEnd
        graph.add_graph(labels, values, "1")
        
        values = [None, None, None, None, self.back_core_bound_stall_on_back(), self.back_memory_bound_stall_on_back(), None, None] # Stalls on BackEnd
        graph.add_graph(labels, values, "1")
        pass
