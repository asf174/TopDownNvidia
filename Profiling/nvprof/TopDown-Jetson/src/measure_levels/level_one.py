"""
Class that represents the level one of the execution

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(1, parentdir) 
from measure_levels.level_execution import LevelExecution
from parameters.level_execution_params import LevelExecutionParameters
from shell.shell import Shell # launch shell arguments
from errors.level_execution_errors import *
from measure_parts.front_end import FrontEnd 
from measure_parts.back_end import BackEnd
from measure_parts.divergence import Divergence
from measure_parts.retire import Retire

class LevelOne(LevelExecution):
    """ 
    Class thath represents the level one of the execution.

    Attributes:
        _front_end      : FrontEnd      ; FrontEnd part of the execution
        _back_end       : BackEnd       ; BackEnd part of the execution
        _divergence     : Divergence    ; Divergence part of the execution
        _retire         : Retire        ; Retire part of the execution
    """

    def __init__(self, program : str, output_file : str):
        self._front_end : FrontEnd = FrontEnd()
        self._back_end  : BackEnd = BackEnd()
        self._divergence : Divergence = Divergence()
        self._retire : Retire = Retire()
        super().__init__(program, output_file)
        pass
 
    def _generate_command(self) -> str:
        """ 
        Generate command of execution with NVIDIA scan tool.

        Returns:
            String with command to be executed
        """

        command : str = ("sudo $(which nvprof) --metrics " + self._front_end.metrics_str() + 
            "," + self._back_end.metrics_str() + "," + self._divergence.metrics_str() + "," + self._extra_measure.metrics_str()
            + "," + self._retire.metrics_str() + "  --events " + self._front_end.events_str() + 
            "," + self._back_end.events_str() + "," + self._divergence.events_str() +  "," + self._extra_measure.events_str() +
             "," + self._retire.events_str() + " --unified-memory-profiling off " + self._program)
        return command
        pass

    def _get_results(self, lst_output : list[str]):
        """
        Get results of the different parts.

        Parameters:
            lst_output              : list[str]     ; OUTPUT list with results
        """

        #  Keep Results
        lst_output.append("\n\nList of counters/metrics measured according to the part.")
        if self._front_end.metrics_str() != "":
            self._add_result_part_to_lst(self._front_end.metrics(), 
                self._front_end.metrics_description(),"\n- FRONT-END RESULTS:", lst_output, True)
        if self._front_end.events_str() != "":
                self._add_result_part_to_lst(self._front_end.events(), 
                self._front_end.events_description(), "", lst_output, False)
        if self._back_end.metrics_str() != "":
            self._add_result_part_to_lst(self._back_end.metrics(), 
                self._back_end.metrics_description(),"\n- BACK-END RESULTS:", lst_output, True)
        if self._back_end.events_str() != "":
                self._add_result_part_to_lst(self._back_end.events(), 
                self._back_end.events_description(), "", lst_output, False)
        if self._divergence.metrics_str() != "":
            self._add_result_part_to_lst(self._divergence.metrics(), 
                self._divergence.metrics_description(),"\n- DIVERGENCE RESULTS:", lst_output, True)
        if self._divergence.events_str() != "":
                self._add_result_part_to_lst(self._divergence.events(), 
                self._divergence.events_description(), "", lst_output, False)
        if self._retire.metrics_str() != "":
                self._add_result_part_to_lst(self._retire.metrics(), 
                self._retire.metrics_description(),"\n- RETIRE RESULTS:", lst_output, True)
        if self._retire.events_str() != "":
                self._add_result_part_to_lst(self._retire.events(), 
                self._retire.events_description(), "", lst_output, False)
        if self._extra_measure.metrics_str() != "":
            self._add_result_part_to_lst(self._extra_measure.metrics(), 
                self._extra_measure.metrics_description(),"\n- EXTRA-MEASURE RESULTS:", lst_output, True)
        if self._extra_measure.events_str() != "":
                self._add_result_part_to_lst(self._extra_measure.events(), 
                self._extra_measure.events_description(), "", lst_output, False)
        lst_output.append("\n")
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

    def _stall_ipc(self) -> float:
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

        return (self.__divergence_ipc_degradation()/self.get_device_max_ipc())*100.0
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
        
        return ((self._stall_ipc()*(self.get_back_end_stall()/100.0))/self.get_device_max_ipc())*100.0
        pass

    def _set_front_back_divergence_retire_results(self, results_launch : str):
        """ Get Results from FrontEnd, BanckEnd, Divergence and Retire parts.
        
        Params:
            results_launch  : str   ; results generated by NVIDIA scan tool
            
        Raises:
            EventNotAsignedToPart       ; raised when an event has not been assigned to any analysis part 
            MetricNotAsignedToPart      ; raised when a metric has not been assigned to any analysis part
        """

        event_name : str
        event_total_value : str 
        metric_name : str
        metric_description : str = ""
        metric_avg_value : str 
        #metric_max_value : str 
        #metric_min_value : str
        has_read_all_events : bool = False
        line : str
        i : int
        list_words : list[str]
        front_end_value_has_found : bool
        frond_end_description_has_found : bool
        back_end_value_has_found : bool
        back_end_description_has_found : bool
        divergence_value_has_found : bool
        divergence_description_has_found : bool
        extra_measure_value_has_found : bool
        extra_measure_description_has_found : bool
        retire_value_has_found : bool 
        retire_description_has_found : bool
        for line in results_launch.splitlines():
            line = re.sub(' +', ' ', line) # delete more than one spaces and put only one
            list_words = line.split(" ")
            if not has_read_all_events:
                # Check if it's line of interest:
                # ['', 'X', 'event_name','Min', 'Max', 'Avg', 'Total'] event_name is str. Rest: numbers (less '', it's an space)
                if len(list_words) > 1: 
                    if list_words[1] == "Metric": # check end events
                        has_read_all_events = True
                    elif list_words[0] == '' and list_words[len(list_words) - 1][0].isnumeric():
                        event_name = list_words[2]
                        event_total_value = list_words[len(list_words) - 1]     
                        front_end_value_has_found = self._front_end.set_event_value(event_name, event_total_value)
                        #frond_end_description_has_found = front_end.set_event_description(event_name, metric_description)
                        back_end_value_has_found = self._back_end.set_event_value(event_name, event_total_value)
                        #back_end_description_has_found = back_end.set_event_description(event_name, metric_description)
                        divergence_value_has_found = self._divergence.set_event_value(event_name, event_total_value)
                        #divergence_description_has_found = divergence.set_event_description(event_name, metric_description)
                        extra_measure_value_has_found = self._extra_measure.set_event_value(event_name, event_total_value)
                        #extra_measure_description_has_found = extra_measure.set_event_description(event_name, metric_description)
                        retire_value_has_found = self._retire.set_event_value(event_name, event_total_value)
                        #retire_description_has_found = extra_measure.set_event_description(event_name, metric_description)
                        if (not (front_end_value_has_found or back_end_value_has_found or divergence_value_has_found or 
                            extra_measure_value_has_found or retire_value_has_found)): #or 
                            #not(frond_end_description_has_found or back_end_description_has_found 
                            #or divergence_description_has_found or extra_measure_description_has_found)):
                            raise EventNotAsignedToPart(event_name)
            else: # metrics
                # Check if it's line of interest:
                # ['', 'X', 'NAME_COUNTER', ... , 'Min', 'Max', 'Avg' (Y%)] where X (int number), Y (int/float number)
                if len(list_words) > 1 and list_words[0] == '' and list_words[len(list_words) - 1][0].isnumeric():
                    metric_name = list_words[2]
                    metric_description = ""
                    for i in range(3, len(list_words) - 3):
                        metric_description += list_words[i] + " "     
                    metric_avg_value = list_words[len(list_words) - 1]
                    #metric_max_value = list_words[len(list_words) - 2]
                    #metric_min_value = list_words[len(list_words) - 3]
                    #if metric_avg_value != metric_max_value or metric_avg_value != metric_min_value:
                        # Do Something. NOT USED
                    front_end_value_has_found = self._front_end.set_metric_value(metric_name, metric_avg_value)
                    frond_end_description_has_found = self._front_end.set_metric_description(metric_name, metric_description)
                    back_end_value_has_found = self._back_end.set_metric_value(metric_name, metric_avg_value)
                    back_end_description_has_found = self._back_end.set_metric_description(metric_name, metric_description)
                    divergence_value_has_found = self._divergence.set_metric_value(metric_name, metric_avg_value)
                    divergence_description_has_found = self._divergence.set_metric_description(metric_name, metric_description)
                    extra_measure_value_has_found = self._extra_measure.set_metric_value(metric_name, metric_avg_value)
                    extra_measure_description_has_found = self._extra_measure.set_metric_description(metric_name, metric_description)
                    retire_value_has_found = self._retire.set_metric_value(metric_name, metric_avg_value)
                    retire_description_has_found = self._retire.set_metric_description(metric_name, metric_description)
                    if (not (front_end_value_has_found or back_end_value_has_found or divergence_value_has_found or 
                        extra_measure_value_has_found or retire_value_has_found) or 
                        not(frond_end_description_has_found or back_end_description_has_found or divergence_description_has_found 
                        or extra_measure_description_has_found or retire_description_has_found)):
                        raise MetricNotAsignedToPart(metric_name)
        pass

    def retire_ipc_percentage(self) -> float:
        """
        Get percentage of TOTAL IPC due to RETIRE.

        Returns:
            Float with percentage of TOTAL IPC due to RETIRE
        """
        return (self.ipc()/self.get_device_max_ipc())*100.0