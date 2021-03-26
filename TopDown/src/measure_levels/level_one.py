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
    def _get_results(self, lst_output : list[str]):
        """
        Get results of the different parts.

        Parameters:
            lst_output              : list[str]     ; OUTPUT list with results
        """
        
        pass

    def run(self, lst_output : list[str]):
        """Run execution."""
        
        output_command : str = super()._launch(self._generate_command())
        self._set_front_back_divergence_retire_results(output_command)
        self._get_results(lst_output)
        pass
    
    def ipc(self) -> float:
        """
        Get IPC of execution.

        Raises:
            IpcMetricNotDefined ; raised if IPC cannot be obtanied because it was not 
            computed by the NVIDIA scan tool.
        """

        ipc_list : list[str] = self._retire.get_metric_value(LevelExecutionParameters.C_IPC_METRIC_NAME)
        if ipc_list is None:
            raise IpcMetricNotDefined # revisar TODO
        total_ipc : float = self._get_total_value_of_list(ipc_list, True)
        return total_ipc
        pass

    def retire_ipc(self) -> float:
        """
        Get "RETIRE" IPC of execution.

        Raises:
            RetireIpcMetricNotDefined ; raised if retire IPC cannot be obtanied because it was not 
            computed by the NVIDIA scan tool.
        """

        warp_execution_efficiency_list : list[str] = self._divergence.get_metric_value(LevelExecutionParameters.C_WARP_EXECUTION_EFFICIENCY_NAME)
        if warp_execution_efficiency_list is None:
            raise RetireIpcMetricNotDefined
        total_warp_execution_efficiency : float = self._get_total_value_of_list(warp_execution_efficiency_list, True)
        return self.ipc()*(total_warp_execution_efficiency/100.0)
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

    def get_front_end_stall(self) -> float:
        """
        Returns percent of stalls due to FrontEnd part.

        Returns:
            Float with percent of total stalls due to FrontEnd
        """

        return self._get_stalls_of_part(self._front_end.metrics())
        pass
    
    def get_back_end_stall(self) -> float:
        """
        Returns percent of stalls due to BackEnd part.

        Returns:
            Float with percent of total stalls due to BackEnd
        """

        return self._get_stalls_of_part(self._back_end.metrics())
        pass

    def _diver_ipc_degradation(self, warp_exec_efficiency_name  : str) -> float:
        """
        Find IPC degradation due to Divergence part based on the name of the required metric.

        Params:
            warp_exec_efficiency_name  : str   ; name of metric to obtain warp execution efficiency

        Returns:
            Float with the Divergence's IPC degradation

        Raises:
            RetireIpcMetricNotDefined ; raised if retire IPC cannot be obtanied because it was not 
            computed by the NVIDIA scan tool.
        """

        ipc : float = self.ipc() 
        warp_execution_efficiency_list  : list[str] = self._divergence.get_metric_value(warp_exec_efficiency_name)
        if warp_execution_efficiency_list is None:
            raise RetireIpcMetricNotDefined # revisar si crear otra excepcion (creo que si)
        total_warp_execution_efficiency : float = self._get_total_value_of_list(warp_execution_efficiency_list, True)
        issued_ipc_list : list[str] = self._divergence.get_metric_value(LevelExecutionParameters.C_ISSUE_IPC_NAME)
        total_issued_ipc : float = self._get_total_value_of_list(issued_ipc_list, True)
        ipc_diference : float = float(total_issued_ipc) - ipc
        if ipc_diference < 0.0:
            ipc_diference = 0.0
        return ipc * (1.0 - (total_warp_execution_efficiency/100.0)) + ipc_diference
        pass

    @abstractmethod
    def __divergence_ipc_degradation(self) -> float:
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

        return super().get_device_max_ipc() - self.retire_ipc() - self.__divergence_ipc_degradation()
        pass

    def divergence_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to Divergence part.

        Returns:
            Float with the percent of Divergence's IPC degradation
        """

        return (self.__divergence_ipc_degradation()/super().get_device_max_ipc())*100.0
        pass

    def front_end_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to FrontEnd part.

        Returns:
            Float with the percent of FrontEnd's IPC degradation
        """
        
        return ((self._stall_ipc()*(self.get_front_end_stall()/100.0))/self.get_device_max_ipc())*100.0
        pass

    def back_end_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to BackEnd part.

        Returns:
            Float with the percent of BackEnd's IPC degradation
        """
        
        return ((self._stall_ipc()*(self.get_back_end_stall()/100.0))/super().get_device_max_ipc())*100.0
        pass

    def retire_ipc_percentage(self) -> float:
        """
        Get percentage of TOTAL IPC due to RETIRE.

        Returns:
            Float with percentage of TOTAL IPC due to RETIRE
        """
        return (self.ipc()/super().get_device_max_ipc())*100.0
