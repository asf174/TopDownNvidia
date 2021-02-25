"""
Class that represents the levels of the execution.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

from abc import ABC, abstractmethod # abstract class
import sys
path : str = "/home/alvaro/Documents/Facultad/"
path_desp : str = "/mnt/HDD/alvaro/"
sys.path.insert(1, path_desp + "TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/errors")
sys.path.insert(1,  path_desp + "TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/parameters")
sys.path.insert(1,  path_desp + "TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/measure_parts")

from measure_parts.extra_measure import ExtraMeasure    
from shell.shell import Shell # launch shell arguments
from parameters.level_execution_params import LevelExecutionParameters # parameters of program
from measure_parts.front_end import FrontEnd 
from measure_parts.back_end import BackEnd
from measure_parts.divergence import Divergence
from measure_parts.retire import Retire
from errors.level_execution_errors import *

class LevelExecution(ABC):
    """ 
    Class that represents the levels of the execution.
     
    Attributes:
        _front_end      : FrontEnd      ; FrontEnd part of the execution
        _back_end       : BackEnd       ; BackEnd part of the execution
        _divergence     : Divergence    ; Divergence part of the execution
        _retire         : Retire        ; Retire part of the execution
        _extra_measure  : ExtraMeasure  ; support measures
        _output_file    : str           ; path to output file with results. 'None' to don't use
                                          output file
        _program        : str           ; program of the execution
    """

    def __init__(self, program : str, output_file : str):
        self._front_end : FrontEnd = FrontEnd()
        self._back_end  : BackEnd = BackEnd()
        self._divergence : Divergence = Divergence()
        self._retire : Retire = Retire()
        self._extra_measure : ExtraMeasure = ExtraMeasure()
        self._program = program
        self._output_file : str = output_file
        pass 

    @abstractmethod
    def run(self, lst_output : list[str]):
        """
        Run execution.
        
        Parameters:
            lst_output  : list[str] ; list with results
        """
        pass

    def ipc(self) -> float:
        """
        Get IPC of execution.

        Raises:
            IpcMetricNotDefined ; raised if IPC cannot be obtanied because it was not 
            computed by the NVIDIA scan tool.
        """

        ipc : str = self._retire.get_metric_value(LevelExecutionParameters.C_IPC_METRIC_NAME)
        warp_execution_efficiency : str = self._divergence.get_metric_value(LevelExecutionParameters.C_WARP_EXECUTION_EFFICIENCY_NAME)
        if ipc is None or warp_execution_efficiency is None:
            raise IpcMetricNotDefined
        return float(ipc)
        pass

    def retire_ipc(self) -> float:
        """
        Get "RETIRE" IPC of execution.

        Raises:
            IpcMetricNotDefined ; raised if IPC cannot be obtanied because it was not 
            computed by the NVIDIA scan tool.
        """

        warp_execution_efficiency : str = self._divergence.get_metric_value(LevelExecutionParameters.C_WARP_EXECUTION_EFFICIENCY_NAME)
        if warp_execution_efficiency is None:
            raise RetireIpcMetricNotDefined
        return self.ipc()*(float(warp_execution_efficiency[0: len(warp_execution_efficiency) - 1])/100)
        pass
    
    def get_device_max_ipc(self) -> float:
        """
        Get Max IPC of device

        Raises:
        """

        shell : Shell = Shell()
        compute_capability : str = shell.launch_command_show_all("nvcc ../src/measure_parts/compute_capability.cu --run", None)
        shell.launch_command("rm -f a.out", None) # delete 'a.out' generated
        if not compute_capability:
            raise ComputeCapabilityError
        dict_warps_schedulers_per_cc : dict = dict({3.0: 4, 3.2: 4, 3.5: 4, 3.7: 4, 5.0: 4, 5.2: 4, 5.3: 4, 
            6.0: 2, 6.1: 4, 6.2: 4, 7.0: 4, 7.5: 4, 8.0: 1}) 
        dict_ins_per_cycle : dict = dict({3.0: 1.5, 3.2: 1.5, 3.5: 1.5, 3.7: 1.5, 5.0: 1.5, 5.2: 1.5, 5.3: 1.5, 
            6.0: 1.5, 6.1: 1.5, 6.2: 1.5, 7.0: 1, 7.5: 1, 8.0: 1})
        return dict_warps_schedulers_per_cc.get(float(compute_capability))*dict_ins_per_cycle.get(float(compute_capability))
        pass
    
    def _add_result_part_to_lst(self, dict_values : dict, dict_desc : dict, message : str, 
        lst_to_add : list[str], isMetric : bool):
        """
        Add results of execution part (FrontEnd, BackEnd...) to list indicated by argument.

        Params:
            dict_values     : dict      ; diccionary with name_metric/event-value elements of the part to 
                                          add to 'lst_to_add'
            dict_desc       : dict      ; diccionary with name_metric/event-description elements of the 
                                          part to add to 'lst_to_add'
            message         : str       ; introductory message to append to 'lst_to_add' to delimit 
                                          the beginning of the region
            lst_output      : list[str] ; list where to add all elements
            isMetric        : bool      ; True if they are metrics or False if they are events

        Raises:
            MetricNoDefined             ; raised in case you have added an metric that is 
                                          not supported or does not exist in the NVIDIA analysis tool
            EventNoDefined              ; raised in case you have added an event that is 
                                          not supported or does not exist in the NVIDIA analysis tool
        """
        
        lst_to_add.append(message)
        #lst_to_add.append("\n")
        lst_to_add.append( "\t\t\t----------------------------------------------------"
            + "---------------------------------------------------")
        if isMetric:
            lst_to_add.append("\t\t\t{:<45} {:<48}  {:<5} ".format('Metric Name','Metric Description', 'Value'))
            lst_to_add.append( "\t\t\t----------------------------------------------------"
            +"---------------------------------------------------")
            for (metric_name, value), (metric_name, desc) in zip(dict_values.items(), 
                dict_desc.items()):
                if metric_name is None or desc is None or value is None:
                    print(str(desc) + str(value) + str(isMetric))
                    raise MetricNoDefined(metric_name)
                lst_to_add.append("\t\t\t{:<45} {:<49} {:<6} ".format(metric_name, desc, value))
        else:
            lst_to_add.append("\t\t\t{:<45} {:<46}  {:<5} ".format('Event Name','Event Description', 'Value'))
            lst_to_add.append( "\t\t\t----------------------------------------------------"
            +"---------------------------------------------------")
            value_event : str 
            for event_name in dict_values:
                value_event = dict_values.get(event_name)
                if event_name is None or value_event is None:
                    #print(str(event_name) + " " + str(value_event))
                    raise EventNoDefined(event_name)
                lst_to_add.append("\t\t\t{:<45} {:<47} {:<6} ".format(event_name, "-", value_event))
        lst_to_add.append("\t\t\t----------------------------------------------------"
            +"---------------------------------------------------")
        pass

    def front_end(self) -> FrontEnd:
        """
        Return FrontEnd part of the execution.

        Returns:
            reference to FrontEnd part of the execution
        """
        return self._front_end
        pass
    
    def back_end(self) -> BackEnd:
        """
        Return BackEnd part of the execution.

        Returns:
            reference to BackEnd part of the execution
        """
        return self._back_end
        pass

    def divergence(self) -> Divergence:
        """
        Return Divergence part of the execution.

        Returns:
            reference to Divergence part of the execution
        """
        return self._divergence
        pass

    def retire(self) -> Retire:
        """
        Return Retire part of the execution.

        Returns:
            reference to Retire part of the execution
        """
        return self._retire
        pass

    def extra_measure(self) -> ExtraMeasure:
        """
        Return ExtraMeasure part of the execution.

        Returns:
            reference to ExtraMeasure part of the execution
        """
        return self._extra_measure
        pass

    def __get__key_max_value(self, dictionary : dict) -> str:
        """ 
        Find key with highest value in dictionary.

        Params:
            dictionary : dict ; dictionary to check

        Returns:
            String with the key with the highest value
            or 'None' if dictionary is empty
        """

        max_key : str = None
        max_value : float = float('-inf')
        value : float
        for key in dictionary.keys():
            value = float(dictionary.get(key)[0 : len(dictionary.get(key)) - 1])
            if value > max_value:
                max_key = key
                max_value = value
        return max_key
        pass

    def get_highest_frond_end_metric(self) -> str:
        """
        Find the highest metric of the FrontEnd Part.

        Returns:
            String with the metric with the highest value
            or 'None' if FrontEnd doesn't have metrics
        """

        if self._front_end.metrics_str() == "":
            return None
        else:
            return self.__get__key_max_value(self._front_end.metrics())
        pass

    def get_highest_frond_end_event(self) -> str:
        """
        Find the highest event of the FrontEnd Part.

        Returns:
            String with the event with the highest value
            or 'None' if FrontEnd doesn't have metrics
        """

        if self._front_end.metrics_str() == "":
            return None
        else:
            return self.__get__key_max_value(self._front_end.metrics())
        pass

    def get_highest_back_end_metric(self) -> str:
        """
        Find the highest metric of the BackEnd Part.

        Returns:
            String with the metric with the highest value
            or 'None' if BackEnd doesn't have metrics
        """

        if self._back_end.metrics_str() == "":
            return None
        else:
            return self.__get__key_max_value(self._back_end.metrics())
        pass

    def get_highest_back_end_event(self) -> str:
        """
        Find the highest event of the BackEnd Part.

        Returns:
            String with the event with the highest value
            or 'None' if BackEnd doesn't have metrics
        """

        if self._back_end.metrics_str() == "":
            return None
        else:
            return self.__get__key_max_value(self._back_end.metrics())
        pass
    
    def get_highest_divergence_metric(self) -> str:
        """
        Find the highest metric of the Divergence Part.

        Returns:
            String with the metric with the highest value
            or 'None' if Divergence doesn't have metrics
        """

        if self._divergence.metrics_str() == "":
            return None
        else:
            return self.__get__key_max_value(self._divergence.metrics())
        pass

    def get_highest_divergence_event(self) -> str:
        """
        Find the highest event of the Divergence Part.

        Returns:
            String with the event with the highest value
            or 'None' if Divergence doesn't have metrics
        """

        if self._divergence.metrics_str() == "":
            return None
        else:
            return self.__get__key_max_value(self._divergence.metrics())
        pass
    
    def get_highest_retire_metric(self) -> str:
        """
        Find the highest metric of the Retire Part.

        Returns:
            String with the metric with the highest value
            or 'None' if Retire doesn't have metrics
        """

        if self._retire.metrics_str() == "":
            return None
        else:
            return self.__get__key_max_value(self._retire.metrics())
        pass

    def get_highest_retire_event(self) -> str:
        """
        Find the highest event of the Retire Part.

        Returns:
            String with the event with the highest value
            or 'None' if Retire doesn't have metrics
        """

        if self._retire.metrics_str() == "":
            return None
        else:
            return self.__get__key_max_value(self._retire.metrics())
        pass

    def get_front_end_stall(self) -> float:
        """
        Returns percent of stalls due to FrontEnd part.

        Returns:
            Float with percent of total stalls due to FrontEnd
        """

        total_value : float = 0.0
        value_str : str 
        for key in self._front_end.metrics().keys():
                value_str = self._front_end.metrics().get(key)
                if value_str[len(value_str) - 1] == "%":  # check percenttage
                    total_value += float(value_str[0 : len(value_str) - 1])
        return round(total_value, 3)
        pass
    
    def get_back_end_stall(self) -> float:
        """
        Returns percent of stalls due to BackEnd part.

        Returns:
            Float with percent of total stalls due to BackEnd
        """

        total_value : float = 0.0
        value_str : str 
        for key in self._back_end.metrics().keys():
                value_str = self._back_end.metrics().get(key)
                if value_str[len(value_str) - 1] == "%":  # check percenttage
                    total_value += float(value_str[0 : len(value_str) - 1])
        return round(total_value, 3)
        pass

    def __divergence_ipc_degradation(self) -> float:
        """
        Find IPC degradation due to Divergence part.

        Returns:
            Float with theDivergence's IPC degradation
        """

        ipc : float = self.ipc() 
        warp_execution_efficiency  : str = self._divergence.get_metric_value(LevelExecutionParameters.C_WARP_EXECUTION_EFFICIENCY_NAME)
        # revisar eleve de excepcion
        issued_ipc : str = self._divergence.get_metric_value(LevelExecutionParameters.C_ISSUE_IPC_NAME)
        if issued_ipc is None:
            raise MetricDivergenceIpcDegradationNotDefined(LevelExecutionParameters.C_ISSUE_IPC_NAME)
        ipc_diference : float = float(issued_ipc) - ipc
        if ipc_diference < 0.0:
            ipc_diference = 0.0
        return ipc * (1.0 - (float(warp_execution_efficiency[0: len(warp_execution_efficiency) - 1])/100.0)) + ipc_diference
        pass

    def __stalls_ipc(self) -> float:
        """
        Find IPC due to STALLS

        Returns:
            Float with STALLS' IPC degradation
        """

        return self.get_device_max_ipc() - self.retire_ipc() - self.__divergence_ipc_degradation()
        pass

    def divergence_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to Divergence part.

        Returns:
            Float with the percent of Divergence's IPC degradation
        """

        return (self.__divergence_ipc_degradation()/(self.get_device_max_ipc()-self.ipc()))*100.0
        pass

    def front_end_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to FrontEnd part.

        Returns:
            Float with the percent of FrontEnd's IPC degradation
        """
        
        return ((self.__stalls_ipc()*(self.get_front_end_stall()/100.0))/(self.get_device_max_ipc()-self.ipc()))*100.0
        pass

    def back_end_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to BackEnd part.

        Returns:
            Float with the percent of BackEnd's IPC degradation
        """
        
        return ((self.__stalls_ipc()*(self.get_back_end_stall()/100.0))/(self.get_device_max_ipc()-self.ipc()))*100.0
        pass
